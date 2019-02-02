# -*- coding: utf-8 -*-
import time

import numpy as np
import six
import tensorflow as tf
from tfsnippet.scaffold import TrainLoop
from tfsnippet.shortcuts import VarScopeObject
from tfsnippet.utils import (reopen_variable_scope,
                             get_default_session_or_error,
                             ensure_variables_initialized,
                             get_variables_as_dict)

from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Trainer']


class Trainer(VarScopeObject):
    """
    OmniAnomaly trainer.

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance.
        model_vs (str or tf.VariableScope): If specified, will collect
            trainable variables only from this scope.  If :obj:`None`,
            will collect all trainable variables within current graph.
            (default :obj:`None`)
        n_z (int or None): Number of `z` samples to take for each `x`.
            (default :obj:`None`, one sample without explicit sampling
            dimension)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            training. (default :obj:`None`, indicating no feeding)
        valid_feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            validation.  If :obj:`None`, follow `feed_dict` of training.
            (default :obj:`None`)
        use_regularization_loss (bool): Whether or not to add regularization
            loss from `tf.GraphKeys.REGULARIZATION_LOSSES` to the training
            loss? (default :obj:`True`)
        max_epoch (int or None): Maximum epochs to run.  If :obj:`None`,
            will not stop at any particular epoch. (default 256)
        max_step (int or None): Maximum steps to run.  If :obj:`None`,
            will not stop at any particular step.  At least one of `max_epoch`
            and `max_step` should be specified. (default :obj:`None`)
        batch_size (int): Size of mini-batches for training. (default 256)
        valid_batch_size (int): Size of mini-batches for validation.
            (default 1024)
        valid_step_freq (int): Run validation after every `valid_step_freq`
            number of training steps. (default 100)
        initial_lr (float): Initial learning rate. (default 0.001)
        lr_anneal_epochs (int): Anneal the learning rate after every
            `lr_anneal_epochs` number of epochs. (default 10)
        lr_anneal_factor (float): Anneal the learning rate with this
            discount factor, i.e., ``learning_rate = learning_rate
            * lr_anneal_factor``. (default 0.75)
        optimizer (Type[tf.train.Optimizer]): The class of TensorFlow
            optimizer. (default :class:`tf.train.AdamOptimizer`)
        optimizer_params (dict[str, any] or None): The named arguments
            for constructing the optimizer. (default :obj:`None`)
        grad_clip_norm (float or None): Clip gradient by this norm.
            If :obj:`None`, disable gradient clip by norm. (default 10.0)
        check_numerics (bool): Whether or not to add TensorFlow assertions
            for numerical issues? (default :obj:`True`)
        name (str): Optional name of this trainer
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this trainer
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, model_vs=None, n_z=None,
                 feed_dict=None, valid_feed_dict=None,
                 use_regularization_loss=True,
                 max_epoch=256, max_step=None, batch_size=256,
                 valid_batch_size=1024, valid_step_freq=100,
                 initial_lr=0.001, lr_anneal_epochs=10, lr_anneal_factor=0.75,
                 optimizer=tf.train.AdamOptimizer, optimizer_params=None,
                 grad_clip_norm=50.0, check_numerics=True,
                 name=None, scope=None):
        super(Trainer, self).__init__(name=name, scope=scope)

        # memorize the arguments
        self._model = model
        self._n_z = n_z
        if feed_dict is not None:
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        if valid_feed_dict is not None:
            self._valid_feed_dict = dict(six.iteritems(valid_feed_dict))
        else:
            self._valid_feed_dict = self._feed_dict
        if max_epoch is None and max_step is None:
            raise ValueError('At least one of `max_epoch` and `max_step` '
                             'should be specified')
        self._max_epoch = max_epoch
        self._max_step = max_step
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size
        self._valid_step_freq = valid_step_freq
        self._initial_lr = initial_lr
        self._lr_anneal_epochs = lr_anneal_epochs
        self._lr_anneal_factor = lr_anneal_factor

        # build the trainer
        with reopen_variable_scope(self.variable_scope):
            # the global step for this model
            self._global_step = tf.get_variable(
                dtype=tf.int64, name='global_step', trainable=False,
                initializer=tf.constant(0, dtype=tf.int64)
            )

            # input placeholders
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.window_length, model.x_dims], name='input_x')
            self._learning_rate = tf.placeholder(
                dtype=tf.float32, shape=(), name='learning_rate')

            # compose the training loss
            with tf.name_scope('loss'):
                loss = model.get_training_loss(
                    x=self._input_x, n_z=n_z)
                if use_regularization_loss:
                    loss += tf.losses.get_regularization_loss()
                self._loss = loss

            # get the training variables
            train_params = get_variables_as_dict(
                scope=model_vs, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            self._train_params = train_params

            # create the trainer
            if optimizer_params is None:
                optimizer_params = {}
            else:
                optimizer_params = dict(six.iteritems(optimizer_params))
            optimizer_params['learning_rate'] = self._learning_rate
            self._optimizer = optimizer(**optimizer_params)

            # derive the training gradient
            origin_grad_vars = self._optimizer.compute_gradients(
                self._loss, list(six.itervalues(self._train_params))
            )
            grad_vars = []
            for grad, var in origin_grad_vars:
                if grad is not None and var is not None:
                    if grad_clip_norm:
                        grad = tf.clip_by_norm(grad, grad_clip_norm)
                    if check_numerics:
                        grad = tf.check_numerics(
                            grad,
                            'gradient for {} has numeric issue'.format(var.name)
                        )
                    grad_vars.append((grad, var))

            # build the training op
            with tf.control_dependencies(
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self._train_op = self._optimizer.apply_gradients(
                    grad_vars, global_step=self._global_step)

            # the training summary in case `summary_dir` is specified
            with tf.name_scope('summary'):
                self._summary_op = tf.summary.merge([
                    tf.summary.histogram(v.name.rsplit(':', 1)[0], v)
                    for v in six.itervalues(self._train_params)
                ])

            # initializer for the variables
            self._trainer_initializer = tf.variables_initializer(
                list(six.itervalues(get_variables_as_dict(scope=self.variable_scope,
                                                          collection=tf.GraphKeys.GLOBAL_VARIABLES)))
            )

    @property
    def model(self):
        """
        Get the :class:`OmniAnomaly` model instance.

        Returns:
            OmniAnomaly: The :class:`OmniAnomaly` model instance.
        """
        return self._model

    def fit(self, values,
            valid_portion=0.3, summary_dir=None):
        """
        Train the :class:`OmniAnomaly` model with given data.

        Args:
            values (np.ndarray): 1-D `float32` array, the standardized
                KPI observations.
            valid_portion (float): Ratio of validation data out of all the
                specified training data. (default 0.3)
            summary_dir (str): Optional summary directory for
                :class:`tf.summary.FileWriter`. (default :obj:`None`,
                summary is disabled)
        """
        sess = get_default_session_or_error()

        # split the training & validation set
        values = np.asarray(values, dtype=np.float32)
        if len(values.shape) != 2:
            raise ValueError('`values` must be a 2-D array')

        n = int(len(values) * valid_portion)
        train_values, v_x = values[:-n], values[-n:]

        train_sliding_window = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.window_length,
            batch_size=self._batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )
        valid_sliding_window = BatchSlidingWindow(
            array_size=len(v_x),
            window_size=self.model.window_length,
            batch_size=self._valid_batch_size,
        )

        # initialize the variables of the trainer, and the model
        sess.run(self._trainer_initializer)
        ensure_variables_initialized(self._train_params)

        # training loop
        lr = self._initial_lr
        with TrainLoop(
                param_vars=self._train_params,
                early_stopping=True,
                summary_dir=summary_dir,
                max_epoch=self._max_epoch,
                max_step=self._max_step) as loop:  # type: TrainLoop
            loop.print_training_summary()

            train_batch_time = []
            valid_batch_time = []

            for epoch in loop.iter_epochs():
                print('train_values:', train_values.shape)
                train_iterator = train_sliding_window.get_iterator([train_values])
                start_time = time.time()
                for step, (batch_x,) in loop.iter_steps(train_iterator):
                    # run a training step
                    start_batch_time = time.time()
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._learning_rate] = lr
                    feed_dict[self._input_x] = batch_x
                    loss, _ = sess.run(
                        [self._loss, self._train_op], feed_dict=feed_dict)
                    loop.collect_metrics({'loss': loss})
                    train_batch_time.append(time.time() - start_batch_time)

                    if step % self._valid_step_freq == 0:
                        train_duration = time.time() - start_time
                        loop.collect_metrics({'train_time': train_duration})
                        # collect variable summaries
                        if summary_dir is not None:
                            loop.add_summary(sess.run(self._summary_op))

                        # do validation in batches
                        with loop.timeit('valid_time'), \
                                loop.metric_collector('valid_loss') as mc:
                            v_it = valid_sliding_window.get_iterator([v_x])
                            for (b_v_x,) in v_it:
                                start_batch_time = time.time()
                                feed_dict = dict(
                                    six.iteritems(self._valid_feed_dict))
                                feed_dict[self._input_x] = b_v_x
                                loss = sess.run(self._loss, feed_dict=feed_dict)
                                valid_batch_time.append(time.time() - start_batch_time)
                                mc.collect(loss, weight=len(b_v_x))

                        # print the logs of recent steps
                        loop.print_logs()
                        start_time = time.time()

                # anneal the learning rate
                if self._lr_anneal_epochs and \
                        epoch % self._lr_anneal_epochs == 0:
                    lr *= self._lr_anneal_factor
                    loop.println('Learning rate decreased to {}'.format(lr),
                                 with_tag=True)

            return {
                'best_valid_loss': float(loop.best_valid_metric),
                'train_time': np.mean(train_batch_time),
                'valid_time': np.mean(valid_batch_time),
            }
