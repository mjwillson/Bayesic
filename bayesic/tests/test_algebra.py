from ..algebra import *

import numpy.testing as npt
import numpy as np
import numpy.random as npr

npr.seed(1234)

# Create some inputs, symbolic ones together with concrete numpy
# arrays:
def randn(*shape):
    return npr.randn(*shape).astype('float32')

# Some square matrices
X = var('X', 2)
Y = var('Y', 2)
X_ = randn(5, 5)
Y_ = randn(5, 5)

# Some vectors
x = var('x', 1)
y = var('y', 1)
x_ = randn(5)
y_ = randn(5)

# 3D tensor
T = var('T', 3)
T_ = randn(3, 5, 7)


def test_add():
    fn = (X + Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ + Y_)


def test_sub():
    fn = (X - Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ - Y_)


def test_abs():
    fn = abs(X).compile()
    npt.assert_allclose(fn(X=X_), abs(X_))


def test_scalar_autobroadcast():
    # this should work on all multi-arg elemwise ops --
    # the scalar gets broadcast up to match the other args
    fn = (X + 1).compile()
    npt.assert_allclose(fn(X=X_), X_ + 1)
    fn = (1 - X).compile()
    npt.assert_allclose(fn(X=X_), 1 - X_)
    # this one will actually go via einsum:
    fn = (2 * X).compile()
    npt.assert_allclose(fn(X=X_), 2 * X_)
    fn = add(1, 1).compile()
    assert fn() == 2


def test_literal_wrapping():
    # this time using a numpy array as a literal
    fn = (X * Y_).compile()
    npt.assert_allclose(fn(X=X_), X_ * Y_)


def test_dot_matrix_matrix():
    # X @ Y should also work on 3.5
    fn = dot(X, Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), np.dot(X_, Y_), rtol=1e-5)


def test_dot_matrix_vector():
    # check dot works as a method too
    fn = X.dot(y).compile()
    npt.assert_allclose(fn(X=X_, y=y_), np.dot(X_, y_), rtol=1e-5)


def test_dot_vector_vector():
    fn = dot(x, y).compile()
    npt.assert_allclose(fn(x=x_, y=y_), np.dot(x_, y_), rtol=1e-5)


def test_mul():
    fn = (X * Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), np.multiply(X_, Y_), rtol=1e-5)


def test_div():
    fn = (X / Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ / Y_)


def test_pow():
    fn = (X ** Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ ** Y_)
    fn = (2 ** X).compile()
    npt.assert_allclose(fn(X=X_), 2 ** X_)
    fn = (X ** 2).compile()
    npt.assert_allclose(fn(X=X_), X_ ** 2)


def test_log_exp_etc():
    fn = log(X).compile()
    npt.assert_allclose(fn(X=X_), np.log(X_))
    fn = exp(X).compile()
    npt.assert_allclose(fn(X=X_), np.exp(X_))


def test_transpose():
    fn = X.T.compile()
    npt.assert_allclose(fn(X=X_), X_.T)


def test_dimshuffle():
    fn = dimshuffle(T, 2, 0, 1).compile()
    npt.assert_allclose(fn(T=T_), np.transpose(T_, (2, 0, 1)))


def test_dimshuffle_broadcasting():
    fn = (X + x.dimshuffle(0, 'x')).compile()
    npt.assert_allclose(fn(X=X_, x=x_), X_ + x_[:, np.newaxis])

    fn = (X * dimshuffle(x, 'x', 0)).compile()
    npt.assert_allclose(fn(X=X_, x=x_), X_ * x_[np.newaxis, :])


def test_trace():
    fn = trace(X).compile()
    npt.assert_allclose(fn(X=X_), np.trace(X_))


def test_diagonal():
    fn = diagonal(X).compile()
    npt.assert_allclose(fn(X=X_), np.diagonal(X_))


def test_outer():
    fn = outer(x, y).compile()
    npt.assert_allclose(fn(x=x_, y=y_), np.outer(x_, y_))


def test_sum():
    fn = sum(T).compile()
    npt.assert_allclose(fn(T=T_), T_.sum(), rtol=1e-5)

    fn = sum(T, axis=0).compile()
    npt.assert_allclose(fn(T=T_), T_.sum(axis=0), rtol=1e-5)

    # Check works as a method too
    fn = T.sum(axis=(0, 2)).compile()
    npt.assert_allclose(fn(T=T_), T_.sum(axis=(0, 2)), rtol=1e-5)


# Test combining of einsums:

def test_composition_of_einsums_collapses_to_single_einsum():
    expr = dot(diagonal(dot(X, outer(x, y))), Y)
    # no einsum parents
    assert expr.parents == [X, x, y, Y]
    # check it works
    npt.assert_allclose(
        expr.compile()(x=x_, y=y_, X=X_, Y=Y_),
        np.dot(np.diagonal(np.dot(X_, np.outer(x_, y_))), Y_),
        rtol=1e-5
    )


def test_two_equivalent_einsum_expressions_same_result():
    # two equivalent expressions:
    expr = trace(dot(X.T, Y))
    expr2 = sum(X * Y)
    assert expr.parents == [X, Y]
    assert expr2.parents == [X, Y]

    # can't always rely on this being the case for algebraically
    # equivalent einsum expressions, since we preserve some
    # information about the order of sums which doesn't matter
    # algebraically. But in this case that's equal too:
    assert expr.factors_and_indices == expr2.factors_and_indices
