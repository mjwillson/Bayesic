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
X = InputExpression('X', 2)
Y = InputExpression('Y', 2)
X_ = randn(5, 5)
Y_ = randn(5, 5)

# Some vectors
x = InputExpression('x', 1)
y = InputExpression('y', 1)
x_ = randn(5)
y_ = randn(5)

# 3D tensor
T = InputExpression('T', 3)
T_ = randn(3, 5, 7)


def test_add():
    fn = add(X, Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ + Y_)


def test_dot_matrix_matrix():
    fn = dot(X, Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), np.dot(X_, Y_), rtol=1e-5)


def test_dot_matrix_vector():
    fn = dot(X, y).compile()
    npt.assert_allclose(fn(X=X_, y=y_), np.dot(X_, y_), rtol=1e-5)


def test_dot_vector_vector():
    fn = dot(x, y).compile()
    npt.assert_allclose(fn(x=x_, y=y_), np.dot(x_, y_), rtol=1e-5)


def test_mul():
    fn = mul(X, Y).compile()
    npt.assert_allclose(fn(X=X_, Y=Y_), np.multiply(X_, Y_), rtol=1e-5)

def test_transpose():
    fn = transpose(X).compile()
    npt.assert_allclose(fn(X=X_), X_.T)

def test_dimshuffle():
    fn = dimshuffle(T, 2, 0, 1).compile()
    npt.assert_allclose(fn(T=T_), np.transpose(T_, (2, 0, 1)))

def test_dimshuffle_broadcasting():
    fn = add(X, dimshuffle(x, 0, 'x')).compile()
    npt.assert_allclose(fn(X=X_, x=x_), X_ + x_[:, np.newaxis])

    fn = mul(X, dimshuffle(x, 'x', 0)).compile()
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

    fn = sum(T, axis=(0, 2)).compile()
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
    expr = trace(dot(transpose(X), Y))
    expr2 = sum(mul(X, Y))
    assert expr.parents == [X, Y]
    assert expr2.parents == [X, Y]

    # can't always rely on this being the case for algebraically
    # equivalent einsum expressions, since we preserve some
    # information about the order of sums which doesn't matter
    # algebraically. But in this case that's equal too:
    assert expr.factors_and_indices == expr2.factors_and_indices
