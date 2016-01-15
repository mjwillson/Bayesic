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

    def log_likelihood(self, data, **params):
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


        # Implementation

        log_likelihood delegates to three different terms, each of
        which should be implemented by subclasses:

        log_likelihood(data, **params) = \
            log_likelihood_data_term(data) \
          + log_likelihood_interaction_term(data, **params) \
          - log_normalizer(data.shape, **params)

        any terms which are independent of the value of the data
        should be split off into log_normalizer; any terms which are
        independent of the value of the parameters should be split off
        into log_likelihood_data_term.

        Note log_normalizer is so called, and is subtracted rather
        than added, because the "normalizer" (the integral over data
        of the rest of the likelihood which makes it normalize) is
        divided by, so the log-normalizer is subtracted.

        Note this representation isn't unique -- it exists mainly to
        enable performance optimisations in cases where the likelihood
        only needs to be computed up to a constant factor in either
        the parameters or the data. The decomoposition is also
        particularly important for ExponentialFamily distributions.


        # Support for tensor of multiple independent observations

        The result will be a scalar if params and data match their
        specified ndim.

        If you want independent_observationsn and iid to work, methods
        for the three log_likelihood terms above must support being
        called with some extra leading dimensions on the shape of both
        params and data for per-observation parameters and data
        respectively.

        It must return a tensor of log-likelihoods one per
        observation, in a corresponding shape to those leading
        dimensions.

        So each param tensor will have shape == obs_shape + per_obs_param_shape,
        the data tensor will have shape == obs_shape + per_obs_data_shape,
        and the result will have shape == obs_shape.

        For scalar expressions this will normally 'just work' based on
        theano broadcasting rules without any extra effort required,
        but if you do anything axis-specific or any special shape
        manipulation (dimshuffle, flatten, reshape etc) you may need
        to be careful.

        """
        raise self.log_likelihood_data_term(data) \
            + self.log_likelihood_interaction_term(data, **params) \
            - self.log_normalizer(data_shape=data.shape, **params)

    def log_normalizer(self, data_shape, **params):
        """The normalization term for the other terms in the
        log-likelihood, which ensures the distribution normalizes
        correctly for each value of the parameters.

        Any terms which depend on the parameters but not the data
        should be incoporated into this term.

        Note it's OK to depend on the shape of the data (unless this
        shape itself has a distribution dependent on the parameters)
        and some distributions need to in their log-normalizers, hence
        the extra argument.

        Should be equal to the log of the integral (or sum, for
        discrete variables) over the data of the rest of the
        likelihood (i.e. the exp of the other two log-likelihood
        terms)

        See log_likelihood for how this decomposition into terms
        works.

                                  """
        raise NotImplementedError

    def log_likelihood_interaction_term(self, data, **params):
        """Any terms in the log-likelihood which depend on both
        parameters and data.

        See log_likelihood for how this decomposition into terms
        works.

        """
        raise NotImplementedError

    def log_likelihood_data_term(self, data):
        """Any terms in the log-likelihood which depend only on the
        data, not the parameters.

        See log_likelihood for how this decomposition into terms
        works.

        """
        raise NotImplementedError

    def independent_observations(self, param_copy_ndim=1, iid_draw_ndim=0):
        """
        Distribution of a tensor of multiple independent observations
        from this distribution.

        Supports multiple copies of the parameters and multiple IID draws
        from each of these copies. (At present the number of IID draws
        must be the same from each copy of the parameters, because
        everything needs to fit into a tensor with fixed dimensions)

        param_copy_ndim -- how many extra leading dimensions will be added
        to both parameters and data to allow for independent draws from
        multiple copies of the parameters. Defaults to 1. If this is 0
        then only a single set of parameters is used.

        iid_draw_ndim -- how many extra leading dimensions (after any
        param_copy dimensions) will be added to the data to support
        multiple IID draws from each copy of the parameters.

        """
        return IndependentObservations(self, param_copy_ndim, iid_draw_ndim)

    def iid(self, extra_ndim=1):
        """Distribution of a tensor of iid draws from this distribution.

        This is a special case of independent_observations"""
        return self.independent_observations(param_copy_ndim=0, iid_draw_ndim=extra_ndim)



class IndependentObservations(ConditionalDistribution):
    """Distribution of a tensor of multiple independent observations
    from another distribution.

    Supports multiple copies of the parameters and multiple IID draws
    from each of these copies. (At present the number of IID draws
    must be the same from each copy of the parameters, because
    everything needs to fit into a tensor with fixed dimensions)

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
        return {
            name: (dtype, self.param_copy_ndim + ndim)
            for name, (dtype, ndim) in self.underlying.parameter_types.items()
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

    def log_normalizer(self, data_shape, **params):
        single_datum_shape = data_shape[self.param_copy_ndim+self.iid_draw_ndim:]
        log_normalizers = self.underlying.log_normalizer(data_shape=single_datum_shape, **params)

        if self.param_copy_ndim > 0:
            sum_of_log_normalizers = normalizers.sum()
        else:
            sum_of_log_normalizers = normalizers

        if self.iid_draw_ndim > 0:
            # We don't need to recompute the normalizer for every IID copy
            # of the data, but we do need to multiply it by the number of
            # iid draws, which requires depending on the shape of the
            # data.
            num_draws_per_parameter = \
                data_shape[self.param_copy_ndim:self.param_copy_ndim+self.iid_draw_ndim].prod()
            return sum_of_log_normalizers * num_draws_per_parameter
        else:
            return sum_of_log_normalizers

    def log_likelihood_interaction_term(self, data, **params):
        broadcasted_params = self._broadcast_params_over_iid_draws(params)
        return self.underlying.log_likelihood_interaction_term(data, **broadcasted_params).sum()

    def log_likelihood_data_term(self, data):
        """Any terms in the log-likelihood which depend only on the
        data, not the parameters.

        See log_likelihood for how this decomposition into terms
        works.

        """
        return self.underlying.log_likelihood_data_term(data).sum()




class ExponentialFamily(ConditionalDistribution):
    """An ExponentialFamily is a special kind of
    ConditionalDistribution where log_likelihood_interaction_term is
    required to be a dot product of some sufficient_statistics of the
    data, and some natural_parameters given in terms of the supplied
    parameters.

    """
    def log_likelihood_interaction_term(self, data, **params):
        """This is implemented by default in terms of a dot product
        between sufficient_statistics and natural_parameters.

        However it may sometimes be more efficient to override this
        with an equivalent but more direct implementation.

        """
        suff_stats = self.sufficient_statistics(data)
        nat_params = self.natural_parameters(**params)

        # may need to be flattened in order to dot product and
        # get a scalar back:
        def flatten(x): return x.flatten() if x.ndim > 1 else x

        # We could concatenate before dot producting, or equivalently
        # sum up component-wise dot products. Choosing the latter as
        # it's a bit more what you'd expect to see in the compute
        # graph.
        terms = [T.dot(flatten(s), flatten(np)) for s, np in zip(suff_stats, nat_params)]
        return sum(terms) if len(terms) > 1 else terms[0]

    def sufficient_statistics(self, data):
        """Should return one or more tensor expressions depending on the data.

        These (flattened and concatted) will be dot producted with the
        natural_parameters (also flattened and concatted) to give the
        log_likelihood_interaction_term.

        Their shapes must match those of the natural_parameters.

        """
        raise NotImplementedError

    def natural_parameters(self, **params):
        """Should return one or more tensor expressions depending on the parameters.

        These (flattened and concatted) will be dot producted with the
        sufficient_statistics (also flattened and concatted) to give the
        log_likelihood_interaction_term.

        Their shapes must match those of the sufficient_statistics.
        """
        raise NotImplementedError

    def independent_observations(self, param_copy_ndim=1, iid_draw_ndim=0):
        """See ConditionalDistribution.independent_observations.

        Note in the case of exponential families, to be supported here
        both sufficient_statistics and natural_parameters should
        support being called with extra leading dimensions,
        per-observation and per-copy-of-parameters respectively.

        """
        return ExpFamIndependentObservations(self, param_copy_ndim, iid_draw_ndim)


class ExpFamIndependentObservations(IndependentObservations):
    def sufficient_statistics(self, data):
        """We can sum up the sufficient statistics for IID draws from the same parameters"""
        iid_draw_dims = tuple(range(self.param_copy_ndim, self.param_copy_ndim + self.iid_draw_ndim))
        return self.underlying.sufficient_statistics(data).sum(axis=iid_draw_dims)

    def natural_parameters(self, **params):
        return self.underlying.natural_parameters(**params).sum(axis=iid_draw_dims)
