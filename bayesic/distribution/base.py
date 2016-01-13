_DISCRETE_DTYPES = ('int8', 'int16', 'int32', 'int64')

class ConditionalDistribution(object):
    """A conditional probability distribution, equivalently a
    parameterized family of distributions.

    """

    @property
    def parameter_types(self):
        """Type metadata for the parameters.

        Dict mapping parameter name to (dtype, ndim)
        """
        raise NotImplementedError

    @property
    def data_type(self):
        """Type metadata for the data, (dtype, ndim)"""
        raise NotImplementedError

    def is_discrete(self):
        return self.data_type in _DISCRETE_DTYPES

    def log_likelihood(self, params, data):
        """Create a symbolic tensor expression for the (normalized)
        log-likelihood of one or more data points, given variables
        for the parameters and the data.

        For discrete distributions this will be a log probability
        mass, for continuous ones a log probability density.

        Params:

        params -- dict mapping parameter names to variables, for each
        parameter in parameter_types, with corresponding dtype and
        ndim (or greater; see below re multiple observations)

        data -- variable with the dtype and ndim given in data_type
        (or greater ndim; see below re multiple observations)

        Returns: theano expression for the log-likelihood.


        == Support for tensor of multiple independent observations

        The result will be a scalar if params and data match their
        specified ndim.

        If they wish to support being wrapped by multiple_observations
        distributions must support being called with some extra
        leading dimensions on the shape of both params and data, and
        must return a tensor of log-likelihoods whose shape is the
        same as these leading dimension.

        So each param tensor will have shape == obs_shape + per_obs_param_shape,
        the data tensor will have shape == obs_shape + per_obs_data_shape,
        and the result will have shape == obs_shape.

        This will normally 'just work' based on theano broadcasting
        rules without any extra effort required.

        See docs on multiple_observations for how to make use of this.

        """
        raise NotImplementedError

    def unnormalized_log_likelihood(self, params, data):
        """As per log_likelihood, but need not be normalized, so any
        terms involving the parameters but not the data can be
        dropped.

        Defaults to calling log_likelihood, but you can override it to
        speed up cases where a normalized likelihood isn't
        required.

        """
        return self.log_likelihood(params, data)

class multiple_observations(ConditionalDistribution):
    """Distribution of a tensor of multiple independent observations
    from another distribution.

    Supports multiple copies of the parameters and multiple IID draws
    from each of these copies.

    param_copy_ndim -- how many extra leading dimensions will be added
    to both parameters and data to allow for independent draws from
    multiple copies of the parameters. Defaults to 1. If this is 0
    then only a single set of parameters is used.

    iid_draw_ndim -- how many extra leading dimensions (after any
    param_copy dimensions) will be added to the data to support
    multiple IID draws from each copy of the parameters.

    """
    def __init__(self, distribution, param_copy_ndim=1, iid_draw_ndim=0):
        self.param_copy_ndim = param_copy_ndim
        self.iid_draw_ndim = iid_draw_ndim
        self.underlying = distribution

    @property
    def parameter_types(self):
        return tuple(
            (dtype, self.param_copy_ndim + ndim)
            for dtype, ndim in self.underlying.parameter_types
        )

    @property
    def data_type(self):
        dtype, ndim = self.underlying.data_type
        return dtype, self.param_copy_ndim + self.iid_draw_ndim + ndim

    def _broadcast_params_over_iid_draws(params):
        def broadcast_over_iid_draws(param, single_obs_param_ndim):
            # Shuffle dims from param_copies_shape x single_obs_param_shape, to
            # param_copies_shape x iid_draws_shape x single_obs_param_shape
            new_dims = tuple(range(self.param_copy_ndim)) + \
                       ('x')*self.iid_draw_dim + \ # extra broadcasting dims
                       tuple(self.param_copy_ndim + d for d in range(single_param_ndim))
            return param.dimshuffle(*new_dims)

        return {
            name: broadcast_over_iid_draws(params[name], ndim)
            for name, (dtype, ndim) in self.parameter_types:
        }

    def log_likelihood(self, params, data):
        broadcasted_params = self._broadcast_params_over_iid_draws(params)
        return self.underlying.log_likelihood(broadcasted_params, data).sum()

    def unnormalized_log_likelihood(self, params, data):
        broadcasted_params = self._broadcast_params_over_iid_draws(params)
        return self.underlying.unnormalized_log_likelihood(broadcasted_params, data).sum()

def iid(distribution, extra_ndim=1):
    """Distribution of a tensor of iid draws from another distribution.

    This is a special case of multiple_observations"""
    return multiple_observations(distribution, param_copy_ndim=0, iid_draw_ndim=extra_ndim)