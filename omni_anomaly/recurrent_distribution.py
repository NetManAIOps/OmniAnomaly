# -*- coding: utf-8 -*-
import tensorflow as tf
from tfsnippet import Distribution, Normal


class RecurrentDistribution(Distribution):
    """
    A multi-variable distribution integrated with recurrent structure.
    """

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_continuous(self):
        return self._is_continuous

    @property
    def is_reparameterized(self):
        return self._is_reparameterized

    @property
    def value_shape(self):
        return self.normal.value_shape

    def get_value_shape(self):
        return self.normal.get_value_shape()

    @property
    def batch_shape(self):
        return self.normal.batch_shape

    def get_batch_shape(self):
        return self.normal.get_batch_shape()

    def sample_step(self, a, t):
        z_previous, mu_q_previous, std_q_previous = a
        noise_n, input_q_n = t
        input_q_n = tf.broadcast_to(input_q_n,
                                    [tf.shape(z_previous)[0], tf.shape(input_q_n)[0], input_q_n.shape[1]])
        input_q = tf.concat([input_q_n, z_previous], axis=-1)
        mu_q = self.mean_q_mlp(input_q, reuse=tf.AUTO_REUSE)  # n_sample * batch_size * z_dim

        std_q = self.std_q_mlp(input_q)  # n_sample * batch_size * z_dim

        temp = tf.einsum('ik,ijk->ijk', noise_n, std_q)  # n_sample * batch_size * z_dim
        mu_q = tf.broadcast_to(mu_q, tf.shape(temp))
        std_q = tf.broadcast_to(std_q, tf.shape(temp))
        z_n = temp + mu_q

        return z_n, mu_q, std_q

    # @global_reuse
    def log_prob_step(self, _, t):

        given_n, input_q_n = t
        if len(given_n.shape) > 2:
            input_q_n = tf.broadcast_to(input_q_n,
                                        [tf.shape(given_n)[0], tf.shape(input_q_n)[0], input_q_n.shape[1]])
        input_q = tf.concat([given_n, input_q_n], axis=-1)
        mu_q = self.mean_q_mlp(input_q, reuse=tf.AUTO_REUSE)

        std_q = self.std_q_mlp(input_q)
        logstd_q = tf.log(std_q)
        precision = tf.exp(-2 * logstd_q)
        if self._check_numerics:
            precision = tf.check_numerics(precision, "precision")
        log_prob_n = - 0.9189385332046727 - logstd_q - 0.5 * precision * tf.square(tf.minimum(tf.abs(given_n - mu_q),
                                                                                              1e8))
        return log_prob_n

    def __init__(self, input_q, mean_q_mlp, std_q_mlp, z_dim, window_length=100, is_reparameterized=True,
                 check_numerics=True):
        super(RecurrentDistribution, self).__init__()
        self.normal = Normal(mean=tf.zeros([window_length, z_dim]), std=tf.ones([window_length, z_dim]))
        self.std_q_mlp = std_q_mlp
        self.mean_q_mlp = mean_q_mlp
        self._check_numerics = check_numerics
        self.input_q = tf.transpose(input_q, [1, 0, 2])
        self._dtype = input_q.dtype
        self._is_reparameterized = is_reparameterized
        self._is_continuous = True
        self.z_dim = z_dim
        self.window_length = window_length
        self.time_first_shape = tf.convert_to_tensor([self.window_length, tf.shape(input_q)[0], self.z_dim])

    def sample(self, n_samples=1024, is_reparameterized=None, group_ndims=0, compute_density=False,
               name=None):

        from tfsnippet.stochastic import StochasticTensor
        if n_samples is None:
            n_samples = 1
            n_samples_is_none = True
        else:
            n_samples_is_none = False
        with tf.name_scope(name=name, default_name='sample'):
            noise = self.normal.sample(n_samples=n_samples)

            noise = tf.transpose(noise, [1, 0, 2])  # window_length * n_samples * z_dim
            noise = tf.truncated_normal(tf.shape(noise))

            time_indices_shape = tf.convert_to_tensor([n_samples, tf.shape(self.input_q)[1], self.z_dim])

            samples = tf.scan(fn=self.sample_step,
                              elems=(noise, self.input_q),
                              initializer=(tf.zeros(time_indices_shape),
                                           tf.zeros(time_indices_shape),
                                           tf.ones(time_indices_shape)),
                              back_prop=False
                              )[0]  # time_step * n_samples * batch_size * z_dim

            samples = tf.transpose(samples, [1, 2, 0, 3])  # n_samples * batch_size * time_step *  z_dim

            if n_samples_is_none:
                t = StochasticTensor(
                    distribution=self,
                    tensor=tf.reduce_mean(samples, axis=0),
                    n_samples=1,
                    group_ndims=group_ndims,
                    is_reparameterized=self.is_reparameterized
                )
            else:
                t = StochasticTensor(
                    distribution=self,
                    tensor=samples,
                    n_samples=n_samples,
                    group_ndims=group_ndims,
                    is_reparameterized=self.is_reparameterized
                )
            if compute_density:
                with tf.name_scope('compute_prob_and_log_prob'):
                    log_p = t.log_prob()
                    t._self_prob = tf.exp(log_p)
            return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='log_prob'):
            if len(given.shape) > 3:
                time_indices_shape = tf.convert_to_tensor([tf.shape(given)[0], tf.shape(self.input_q)[1], self.z_dim])
                given = tf.transpose(given, [2, 0, 1, 3])
            else:
                time_indices_shape = tf.convert_to_tensor([tf.shape(self.input_q)[1], self.z_dim])
                given = tf.transpose(given, [1, 0, 2])
            log_prob = tf.scan(fn=self.log_prob_step,
                               elems=(given, self.input_q),
                               initializer=tf.zeros(time_indices_shape),
                               back_prop=False
                               )
            if len(given.shape) > 3:
                log_prob = tf.transpose(log_prob, [1, 2, 0, 3])
            else:
                log_prob = tf.transpose(log_prob, [1, 0, 2])

            if group_ndims == 1:
                log_prob = tf.reduce_sum(log_prob, axis=-1)
            return log_prob

    def prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name, default_name='prob'):
            log_prob = self.log_prob(given, group_ndims, name)
            return tf.exp(log_prob)
