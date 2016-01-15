"""Simple core distributions"""

from bayesic.theano_utils import floatX
from bayesic.distributions.base import ExponentialFamily

from theano import tensor as T

class Normal(ExponentialFamily):
    """A normal or Gaussian distribution given in terms of its mean and variance."""

    parameter_types = dict(
        mean = (floatX, 0),
        variance = (floatX, 0)
    )

    def sufficient_statistics(self, data):
        return data, data**2

    def natural_parameters(self, mean, variance):
        return mean / variance, -0.5 / variance

    def log_normalizer(self, mean, variance, data_shape=None):
        return -0.5 * np.log(2 * np.pi) \
             + 0.5 * T.log(variance) \
             + 0.5 * (mean / variance)**2

    def log_likelihood_data_term(self, data):
        return 0

class MultivariateNormal(ExponentialFamily):
    """A multivariante Normal given in terms of its mean and precision
    matrix (inverse of the covariance matrix)

    """

    parameter_types = dict(
        mean = (floatX, 1),
        precision = (floatX, 2)
    )

    def sufficient_statistics(self, data):
        # TODO: summing outer products isn't very efficient for iid observations.
        # implement optimsed summed_sufficient_statistics using matrix product.
        return mean, T.outer(mean, mean)

    def natural_parameters(self, mean):
        return T.dot(precision, mean), -0.5 * precision

    def log_normalizer(self, mean, variance, data_shape):
        return -0.5 * data_shape[0] * np.log(2 * np.pi) \
             - 0.5 * T.logdet(precision) \
             + 0.5 * T.dot(mean, T.dot(precision, mean))

    def log_likelihood_data_term(self, data):
        return 0
