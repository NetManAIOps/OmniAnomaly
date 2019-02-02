# -*- coding: utf-8 -*-
import tensorflow as tf
from tfsnippet.bayes import BayesianNet
from tfsnippet.distributions import Distribution
from tfsnippet.stochastic import StochasticTensor, validate_n_samples_arg
from tfsnippet.utils import (instance_reuse, is_tensor_object,
                             reopen_variable_scope, VarScopeObject)
from tfsnippet.variational import VariationalChain


class VAE(VarScopeObject):
    """
    A general implementation of variational auto-encoder as module.

    The variational auto-encoder ("Auto-Encoding Variational Bayes",
    Kingma, D.P. and Welling) is a deep Bayesian network, with observed
    variable `x` and latent variable `z`.  The generative process
    starts from `z` with prior distribution :math:`p(z)`, following a
    hidden network :math:`h(z)`, then comes to `x` with distribution
    :math:`p(x|h(z))`.  To do posterior inference of :math:`p(z|x)`,
    variational inference techniques are adopted, to train a separated
    distribution :math:`q(z|h(x))` (:math:`h(x)` denoting the hidden network)
    to approximate :math:`p(z|x)`.

    This class provides a general implementation of variational auto-encoder,
    with customizable :math:`p(z)`, :math:`p(x|h(z))`, :math:`q(z|h(x))`,
    as well as the hidden networks :math:`h(z)` and :math:`h(x)`.

    For example, to construct a VAE with diagonal Normal `z` and `x`:

    .. code-block:: python

        from tensorflow import keras as K
        from tfsnippet.modules import VAE, DictMapper, Sequential
        from tfsnippet.distributions import Normal

        batch_size = 128
        x_dims, z_dims = 100, 10
        vae = VAE(
            p_z=Normal(mean=tf.zeros([batch_size, z_dims]),
                       std=tf.ones([batch_size, x_dims])),
            p_x_given_z=Normal,
            q_z_given_x=Normal,
            h_for_p_x=Sequential([
                K.layers.Dense(100, activation=tf.nn.relu),
                DictMapper({'mean': K.layers.Dense(x_dims),
                            'logstd': K.layers.Dense(x_dims)})
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(100, activation=tf.nn.relu),
                DictMapper({'mean': K.layers.Dense(z_dims),
                            'logstd': K.layers.Dense(z_dims)})
            ])
        )

    To train the `vae`:

    .. code-block:: python

        # Automatically derive a single-sample loss.
        # Depending on ``z.is_reparameterized``, it might be derived by
        # `sgvb` (is_reparameterized == True) or `reinforce` (otherwise).
        loss = vae.get_training_loss(x)

        # Automatically derive a multi-sample loss.
        # Depending on ``z.is_reparameterized``, it might be derived by
        # `iwae` (is_reparameterized == True) or `vimco` (otherwise).
        loss = vae.get_training_loss(x, n_z=10)

        # Or manually derive a reweighted wake-sleep training loss.
        # Note the `VariationalTrainingObjectives` produce per-data
        # training objectives, instead of a 0-d scalar loss as the
        # `VAE.get_training_loss` does.
        chain = vae.chain(x, n_z=10)
        loss = tf.reduce_mean(chain.vi.training.rws_wake())

    To map from `x` to `z`:

    .. code-block:: python

        # use the :class:`Module` interface for one-to-one mapping
        z = vae(x)

        # use the :class:`Module` interface for multiple `z` samples
        z = vae(x, n_z=10)

        # or obtain the variational :class:`BayesianNet` with observed `z`
        q_net = vae.variational(x, z=observed_z)
        z_log_prob = q_net['z'].log_prob()

    To reconstruct `x`:

    .. code-block:: python

        # use the :meth:`VAE.reconstruct` for obtaining one `x` sample
        x_reconstructed = vae.reconstruct(x)

        # to obtain multiple `z` samples, and further multiple `x` samples
        # (this results in 100 `x` samples for each input `x`)
        x_reconstructed = vae.reconstruct(x, n_z=10, n_x=10)

    To sample `x` from prior `z` or observed `z`:

    .. code-block:: python

        # sample multiple prior `z`, then one `x` for each `z`
        x = vae.model(n_z=10)['x']

        # sample multiple `x` for each observed `z`
        x = vae.model(z=observed_z, n_x=10)['x']
    """

    def __init__(self, p_z, p_x_given_z, q_z_given_x, h_for_p_x, h_for_q_z,
                 z_group_ndims=1, x_group_ndims=1, is_reparameterized=None,
                 name=None, scope=None):
        """
        Construct the :class:`VAE`.

        Args:
            p_z (Distribution): :math:`p(z)`, the distribution instance.
            p_x_given_z: :math:`p(x|h(z))`, a distribution class or
                a :class:`DistributionFactory` object.
            q_z_given_x: :math:`q(z|h(x))`, a distribution class or
                a :class:`DistributionFactory` object.
            h_for_p_x (Module): :math:`h(z)`, the hidden network module for
                :math:`p(x|h(z))`. The output of `h_for_p_x` must be a
                ``dict[str, any]``, the parameters for `p_x_given_z`.
            h_for_q_z (Module): :math:`h(x)`, the hidden network module for
                :math:`q(z|h(x))`. The output of `h_for_q_z` must be a
                ``dict[str, any]``, the parameters for `q_z_given_x`.
            z_group_ndims (int or tf.Tensor): `group_ndims` for `z`. (default 1)
            x_group_ndims (int or tf.Tensor): `group_ndims` for `x`. (default 1)
            is_reparameterized (bool or None): Whether or not `z` should be
                re-parameterized? (default :obj:`None`, following the settings
                of z distributions.)
            name (str): Optional name of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).

        See Also:
            :meth:`tfsnippet.distributions.Distribution.log_prob` for
                contents about `group_ndims`.
        """
        if not isinstance(p_z, Distribution):
            raise TypeError('`p_z` must be an instance of `Distribution`')
        if not callable(h_for_p_x):
            raise TypeError('`h_for_p_x` must be an instance of `Module` or '
                            'a callable object')
        if not callable(h_for_q_z):
            raise TypeError('`h_for_q_z` must be an instance of `Module` or '
                            'a callable object')
        super(VAE, self).__init__(name=name, scope=scope)

        # Defensive coding: wrap `h_for_p_x` and `h_for_q_z` in reused scope.
        if not isinstance(h_for_p_x, VarScopeObject):
            with reopen_variable_scope(self.variable_scope):
                h_for_p_x = Lambda(h_for_p_x, name='h_for_p_x')
        if not isinstance(h_for_q_z, VarScopeObject):
            with reopen_variable_scope(self.variable_scope):
                h_for_q_z = Lambda(h_for_q_z, name='h_for_q_z')

        self._p_z = p_z
        self._p_x_given_z = p_x_given_z
        self._q_z_given_x = q_z_given_x
        self._h_for_p_x = h_for_p_x
        self._h_for_q_z = h_for_q_z
        self._z_group_ndims = z_group_ndims
        self._x_group_ndims = x_group_ndims
        self._is_reparameterized = is_reparameterized

    def __call__(self, inputs, **kwargs):
        with reopen_variable_scope(self.variable_scope):
            # Here `reopen_name_scope` is set to True, so that multiple
            # calls to the same Module instance will always generate operations
            # within the original name scope.
            # However, in order for ``tf.variable_scope(default_name=...)``
            # to work properly with variable reusing, we must generate a nested
            # unique name scope.
            with tf.name_scope('forward'):
                return self._forward(inputs, **kwargs)

    @property
    def p_z(self):
        """
        Get :math:`p(z)`, the prior distribution of `z`.

        Returns:
            Distribution: The distribution instance.
        """
        return self._p_z

    @property
    def p_x_given_z(self):
        """
        Get the factory for :math:`p(x|h(z))`.

        Returns:
            DistributionFactory: The distribution factory.
        """
        return self._p_x_given_z

    @property
    def q_z_given_x(self):
        """
        Get the factory for :math:`q(z|h(x))`.

        Returns:
            DistributionFactory: The distribution factory.
        """
        return self._q_z_given_x

    @property
    def h_for_p_x(self):
        """
        Get :math:`h(z)`, the hidden network for :math:`p(x|h(z))`.

        Returns:
            Module: The hidden network.
        """
        return self._h_for_p_x

    @property
    def h_for_q_z(self):
        """
        Get :math:`h(x)`, the hidden network for :math:`q(z|h(x))`.

        Returns:
            Module: The hidden network.
        """
        return self._h_for_q_z

    @property
    def z_group_ndims(self):
        """Get the `group_ndims` for `z`."""
        return self._z_group_ndims

    @property
    def x_group_ndims(self):
        """Get the `group_ndims` for `x`."""
        return self._x_group_ndims

    @property
    def is_reparameterized(self):
        """Whether or not `z` is re-parameterized?"""
        return self._is_reparameterized

    @instance_reuse
    def variational(self, x, z=None, n_z=None, posterior_flow=None):
        """
        Derive an instance of :math:`q(z|h(x))`, the variational net.

        Args:
            x: The observation `x` for the variational net.
            z: If specified, observe `z` in the variational net.
                (default :obj:`None`)
            n_z: The number of `z` samples to take for each `x`, if `z`
                is not observed. (default :obj:`None`, one sample for
                each `x`, without dedicated sampling dimension)

                It is recommended to specify this argument even if `z`
                is observed, to make explicit how many samples are there
                in the observation.

        Returns:
            BayesianNet: The variational net.
        """
        observed = {}
        if z is not None:
            observed['z'] = z
        net = BayesianNet(observed=observed)
        with tf.variable_scope('h_for_q_z'):
            z_params = self.h_for_q_z(x)
        with tf.variable_scope('q_z_given_x'):
            q_z_given_x = self.q_z_given_x(**z_params)
            assert (isinstance(q_z_given_x, Distribution))
        with tf.name_scope('z'):
            z = net.add('z', q_z_given_x, n_samples=n_z,
                        group_ndims=self.z_group_ndims,
                        is_reparameterized=self.is_reparameterized,
                        flow=posterior_flow)
        return net

    @instance_reuse
    def model(self, z=None, x=None, n_z=None, n_x=None):
        """
        Derive an instance of :math:`p(x|h(z))`, the model net.

        Args:
            z: If specified, observe `z` in the model net. (default :obj:`None`)
            x: If specified, observe `x` in the model net. (default :obj:`None`)
            n_z: The number of `z` samples to take for each `x`, if `z`
                is not observed. (default :obj:`None`, one `z` sample for
                each `x`, without dedicated sampling dimension)

                It is recommended to specify this argument even if `z`
                is observed, to make explicit how many samples are there
                in the observation.
            n_x: The number of `x` samples to take for each `z`, if `x`
                is not observed. (default :obj:`None`, one `x` sample for
                each `z`, without dedicated sampling dimension)

                It is recommended to specify this argument even if `x`
                is observed, to make explicit how many samples are there
                in the observation.

        Returns:
            BayesianNet: The variational net.
        """
        observed = {k: v for k, v in [('z', z), ('x', x)] if v is not None}
        net = BayesianNet(observed=observed)
        with tf.name_scope('z'):
            z = net.add('z', self.p_z, n_samples=n_z,
                        group_ndims=self.z_group_ndims,
                        is_reparameterized=self.is_reparameterized)
        with tf.variable_scope('h_for_p_x'):
            x_params = self.h_for_p_x(z)
        with tf.variable_scope('p_x_given_z'):
            p_x_given_z = self.p_x_given_z(**x_params)
            assert (isinstance(p_x_given_z, Distribution))
        with tf.name_scope('x'):
            x = net.add('x', p_x_given_z, n_samples=n_x,
                        group_ndims=self.x_group_ndims)
        return net

    def chain(self, x, n_z=None, posterior_flow=None):
        """
        Chain :math:`q(z|h(x))` and :math:`p(x,z|h(x))` together.

        This method chains the variational net :math:`q(z|h(x))` and the
        model net :math:`p(x,z|h(x))` together, with specified observation
        `x`.  It is typically used to derive the training objectives of VAE.
        It can also be used to calculate the `reconstruction probability`
        ("Variational Autoencoder based Anomaly Detection using Reconstruction
        Probability", An, J. and Cho, S. 2015) of `x`.

        Notes:
            The constructed :class:`~tfsnippet.variational.VariationalChain`
            have `x` observed in its `model` net, thus this method cannot
            be used to get reconstructed samples.  Use :meth:`reconstruct`
            instead to obtain `x` samples.

        Args:
            x: The input observation `x`.
            n_z: Number of `z` samples to take. (default :obj:`None`)

        Returns:
            VariationalChain: The variational chain.
        """
        with tf.name_scope('VAE.chain'):
            q_net = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)

            # automatically detect the `latent_axis` for this chain
            if n_z is not None:
                latent_axis = 0
            else:
                latent_axis = None

            chain = q_net.variational_chain(
                lambda observed: self.model(n_z=n_z, n_x=None, **observed),
                latent_axis=latent_axis,
                observed={'x': x}
            )
        return chain

    def get_training_loss(self, x, n_z=None):
        """
        Get the training loss for this VAE.

        The variational solver is automatically chosen according to
        `z.is_reparameterized`, and the argument `n_z`, by the following rules:

        1. If `z.is_reparameterized` is :obj:`True`, then:

            1. If `n_z` > 1, use `iwae`.
            2. If `n_z` == 1 or `n_z` is :obj:`None`, use `sgvb`.

        2. If `z.is_reparameterized` is :obj:`False`, then:

            1. If `n_z` > 1, use `vimco`.
            2. If `n_z` == 1 or `n_z` is :obj:`None`, use `reinforce`.

        Dynamic `n_z` is not supported by this method.  Also, Reweighted
        Wake-Sleep algorithm is not a choice of this method.  To derive
        the training loss for either situation, use :meth:`chain`
        to obtain a :class:`~tfsnippet.variational.VariationalChain`,
        and further obtain the loss by `chain.vi.training.[algorithm]`.

        Args:
            x: The input observation `x`.
            n_z (int or None): Number of `z` samples to take.  Must be
                :obj:`None` or a constant integer.  Dynamic tensors are not
                accepted, since we cannot automatically choose a variational
                solver for undeterministic `n_z`. (default :obj:`None`)

        Returns:
            tf.Tensor: A 0-d tensor, the training loss which can be optimized
                by gradient descent.

        See Also:
            :class:`tfsnippet.variational.VariationalChain`,
            :class:`tfsnippet.variational.VariationalTrainingObjectives`
        """
        with tf.name_scope('VAE.get_training_loss'):
            if n_z is not None:
                if is_tensor_object(n_z):
                    raise TypeError('Cannot choose the variational solver '
                                    'automatically for dynamic `n_z`')
                n_z = validate_n_samples_arg(n_z, 'n_z')

            # derive the variational chain
            chain = self.chain(x, n_z)
            z = chain.variational['z']

            # auto choose a variational solver for training loss
            if n_z is not None and n_z > 1:
                if z.is_reparameterized:
                    solver = chain.vi.training.iwae
                else:
                    solver = chain.vi.training.vimco
            else:
                if z.is_reparameterized:
                    solver = chain.vi.training.sgvb
                else:
                    solver = chain.vi.training.reinforce

            # derive the training loss
            return tf.reduce_mean(solver())

    def reconstruct(self, x, n_z=None, n_x=None, posterior_flow=None):
        """
        Sample reconstructed `x` from :math:`p(x|h(z))`, where `z` is (are)
        sampled from :math:`q(z|h(x))` using the specified observation `x`.

        Args:
            x: The observation `x` for :math:`q(z|h(x))`.
            n_z: Number of intermediate `z` samples to take for each input `x`.
            n_x: Number of reconstructed `x` samples to take for each `z`.

        Returns:
            StochasticTensor: The reconstructed samples `x`.
        """
        with tf.name_scope('VAE.reconstruct'):
            q_net = self.variational(x, n_z=n_z, posterior_flow=posterior_flow)
            model = self.model(z=q_net['z'], n_z=n_z, n_x=n_x)
            return model['x']

    def _forward(self, inputs, n_z=None, **kwargs):
        """
        Get a `z` sample from :math:`q(z|h(x))`, using the variational net.

        Args:
            inputs: The input `x`.
            n_z: Number of samples to taken for `z`. (default :obj:`None`)
            \**kwargs: Capturing and ignoring all other parameters.  This is
                the default behavior of a :class:`Module`.

        Returns:
            StochasticTensor: The `z` samples.
        """
        q_net = self.variational(inputs, z=None, n_z=n_z, **kwargs)
        return q_net['z']


class Lambda(VarScopeObject):
    """
    Wrapping arbitrary function into a neural network :class:`Module`.

    This class wraps an arbitrary function or lambda expression into
    a neural network :class:`Module`, reusing the variables created
    within the specified function.

    For example, one may wrap :func:`tensorflow.contrib.layers.fully_connected`
    into a reusable module with :class:`Lambda` component as follows:

    .. code-block:: python

        import functools
        from tensorflow.contrib import layers

        dense = Lambda(
            functools.partial(
                layers.fully_connected,
                num_outputs=100,
                activation_fn=tf.nn.relu
            )
        )
    """

    def __init__(self, f, name=None, scope=None):
        """
        Construct the :class:`Lambda`.

        Args:
            f ((inputs, \**kwargs) -> outputs): The function or lambda
                expression which derives the outputs.
            name (str): Optional name of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Optional scope of this module
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        super(Lambda, self).__init__(name=name, scope=scope)
        self._factory = f

    def _forward(self, inputs, **kwargs):
        return self._factory(inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        with reopen_variable_scope(self.variable_scope):
            # Here `reopen_name_scope` is set to True, so that multiple
            # calls to the same Module instance will always generate operations
            # within the original name scope.
            # However, in order for ``tf.variable_scope(default_name=...)``
            # to work properly with variable reusing, we must generate a nested
            # unique name scope.
            with tf.name_scope('forward'):
                return self._forward(inputs, **kwargs)

