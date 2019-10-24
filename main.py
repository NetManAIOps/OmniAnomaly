# -*- coding: utf-8 -*-
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from pprint import pformat, pprint

import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z


class ExpConfig(Config):
    # dataset configuration
    dataset = "machine-1-1"
    x_dim = get_data_dim(dataset)

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 10
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 50
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.

    early_stop = True  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.01

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = 'model'
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = 'result'  # Where to save the result file
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'


def main():
    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # prepare the data
    (x_train, _), (x_test, y_test) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)

    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # construct the trainer
        trainer = Trainer(model=model,
                          model_vs=model_vs,
                          max_epoch=config.max_epoch,
                          batch_size=config.batch_size,
                          valid_batch_size=config.test_batch_size,
                          initial_lr=config.initial_lr,
                          lr_anneal_epochs=config.lr_anneal_epoch_freq,
                          lr_anneal_factor=config.lr_anneal_factor,
                          grad_clip_norm=config.gradient_clip_norm,
                          valid_step_freq=config.valid_step_freq)

        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():

            if config.restore_dir is not None:
                # Restore variables from `save_dir`.
                saver = VariableSaver(get_variables_as_dict(model_vs), config.restore_dir)
                saver.restore()

            if config.max_epoch > 0:
                # train the model
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train)
                train_time = (time.time() - train_start) / config.max_epoch
                best_valid_metrics.update({
                    'train_time': train_time
                })
            else:
                best_valid_metrics = {}

            # get score of train set for POT algorithm
            train_score, train_z, train_pred_speed = predictor.get_score(x_train)
            if config.train_score_filename is not None:
                with open(os.path.join(config.result_dir, config.train_score_filename), 'wb') as file:
                    pickle.dump(train_score, file)
            if config.save_z:
                save_z(train_z, 'train_z')

            if x_test is not None:
                # get score of test set
                test_start = time.time()
                test_score, test_z, pred_speed = predictor.get_score(x_test)
                test_time = time.time() - test_start
                if config.save_z:
                    save_z(test_z, 'test_z')
                best_valid_metrics.update({
                    'pred_time': pred_speed,
                    'pred_total_time': test_time
                })
                if config.test_score_filename is not None:
                    with open(os.path.join(config.result_dir, config.test_score_filename), 'wb') as file:
                        pickle.dump(test_score, file)

                if y_test is not None and len(y_test) >= len(test_score):
                    if config.get_score_on_dim:
                        # get the joint score
                        test_score = np.sum(test_score, axis=-1)
                        train_score = np.sum(train_score, axis=-1)

                    # get best f1
                    t, th = bf_search(test_score, y_test[-len(test_score):],
                                      start=config.bf_search_min,
                                      end=config.bf_search_max,
                                      step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                                                   config.bf_search_step_size),
                                      display_freq=50)
                    # get pot results
                    pot_result = pot_eval(train_score, test_score, y_test[-len(test_score):], level=config.level)

                    # output the results
                    best_valid_metrics.update({
                        'best-f1': t[0],
                        'precision': t[1],
                        'recall': t[2],
                        'TP': t[3],
                        'TN': t[4],
                        'FP': t[5],
                        'FN': t[6],
                        'latency': t[-1],
                        'threshold': th
                    })
                    best_valid_metrics.update(pot_result)
                results.update_metrics(best_valid_metrics)

            if config.save_dir is not None:
                # save the variables
                var_dict = get_variables_as_dict(model_vs)
                saver = VariableSaver(var_dict, config.save_dir)
                saver.save()
            print('=' * 30 + 'result' + '=' * 30)
            pprint(best_valid_metrics)


if __name__ == '__main__':

    # get config obj
    config = ExpConfig()

    # parse the arguments
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])
    config.x_dim = get_data_dim(config.dataset)

    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories if specified
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs(config.save_dir, exist_ok=True)
    with warnings.catch_warnings():
        # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
        main()
