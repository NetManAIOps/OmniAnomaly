# -*- coding: utf-8 -*-
import time

import numpy as np
import six
import tensorflow as tf
from tfsnippet.utils import (VarScopeObject, get_default_session_or_error,
                             reopen_variable_scope)

from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Predictor']


class Predictor(VarScopeObject):
    """
    OmniAnomaly predictor.

    Args:
        model (OmniAnomaly): The :class:`OmniAnomaly` model instance.
        n_z (int or None): Number of `z` samples to take for each `x`.
            If :obj:`None`, one sample without explicit sampling dimension.
            (default 1024)
        batch_size (int): Size of each mini-batch for prediction.
            (default 32)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            prediction. (default :obj:`None`)
        last_point_only (bool): Whether to obtain the reconstruction
            probability of only the last point in each window?
            (default :obj:`True`)
        name (str): Optional name of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, n_z=1024, batch_size=32,
                 feed_dict=None, last_point_only=True, name=None, scope=None):
        super(Predictor, self).__init__(name=name, scope=scope)
        self._model = model
        self._n_z = n_z
        self._batch_size = batch_size
        if feed_dict is not None:
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        self._last_point_only = last_point_only

        with reopen_variable_scope(self.variable_scope):
            # input placeholders
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.window_length, model.x_dims], name='input_x')
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.window_length], name='input_y')

            # outputs of interest
            self._score = self._score_without_y = None

    def _get_score_without_y(self):
        if self._score_without_y is None:
            with reopen_variable_scope(self.variable_scope), \
                 tf.name_scope('score_without_y'):
                self._score_without_y, self._q_net_z = self.model.get_score(
                    x=self._input_x,
                    n_z=self._n_z,
                    last_point_only=self._last_point_only
                )
                # print ('\t_get_score_without_y ',type(self._q_net_z))
        return self._score_without_y, self._q_net_z

    @property
    def model(self):
        """
        Get the :class:`OmniAnomaly` model instance.

        Returns:
            OmniAnomaly: The :class:`OmniAnomaly` model instance.
        """
        return self._model

    def get_score(self, values):
        """
        Get the `reconstruction probability` of specified KPI observations.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            values (np.ndarray): 1-D float32 array, the KPI observations.

        Returns:
            np.ndarray: The `reconstruction probability`,
                1-D array if `last_point_only` is :obj:`True`,
                or 2-D array if `last_point_only` is :obj:`False`.
        """
        with tf.name_scope('Predictor.get_score'):
            sess = get_default_session_or_error()
            collector = []
            collector_z = []

            # validate the arguments
            values = np.asarray(values, dtype=np.float32)
            if len(values.shape) != 2:
                raise ValueError('`values` must be a 2-D array')

            # run the prediction in mini-batches
            sliding_window = BatchSlidingWindow(
                array_size=len(values),
                window_size=self.model.window_length,
                batch_size=self._batch_size,
            )

            pred_time = []

            for b_x, in sliding_window.get_iterator([values]):
                start_iter_time = time.time()
                feed_dict = dict(six.iteritems(self._feed_dict))
                feed_dict[self._input_x] = b_x
                b_r, q_net_z = sess.run(self._get_score_without_y(),
                                        feed_dict=feed_dict)
                collector.append(b_r)
                pred_time.append(time.time() - start_iter_time)
                collector_z.append(q_net_z)

            # merge the results of mini-batches
            result = np.concatenate(collector, axis=0)
            result_z = np.concatenate(collector_z, axis=0)
            return result, result_z, np.mean(pred_time)
