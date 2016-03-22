from ..algebra import *

import numpy.testing as npt
import numpy as np
import numpy.random as npr

from nose_parameterized import parameterized

npr.seed(1234)

# Create some inputs, symbolic ones together with concrete numpy
# arrays:
def randn(*shape):
    return npr.randn(*shape).astype('float32')

# Some square matrices
X = var('X', 2)
Y = var('Y', 2)
Z = var('Z', 2)
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

# int scalar
a = var('a', 0, 'int32')


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
    npt.assert_allclose(fn(X=X_, Y=Y_), X_ / Y_, rtol=1e-5)


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


def test_shape_and_size():
    fn = X.shape[0].compile()
    assert fn(X=[[1,2],[3,4],[5,6]]) == 3
    fn = X.shape[1].compile()
    assert fn(X=[[1,2],[3,4],[5,6]]) == 2
    fn = X.size.compile()
    assert fn(X=[[1,2],[3,4],[5,6]]) == 6


def test_eye():
    fn = eye(a).compile()
    npt.assert_equal(fn(a=2), np.eye(2))
    npt.assert_equal(fn(a=5), np.eye(5))


def test_composition_of_einsums_collapses_to_single_einsum():
    expr = dot(diagonal(dot(X, outer(x, y))), Y)
    # no einsum parents
    assert expr.parents == (X, x, y, Y)
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
    assert expr.parents == (X, Y)
    assert expr2.parents == (X, Y)

    # can't always rely on this being the case for algebraically
    # equivalent einsum expressions, since we preserve some
    # information about the order of sums which doesn't matter
    # algebraically. But in this case that's equal too:
    assert expr.factors_and_indices == expr2.factors_and_indices


@parameterized([
    ([1], []),
    ([1, 1], [1]),
    ([1, 1, 2], [1, 2])
])
def test_find_injections_none_possible(A, B):
    assert list(find_injections(A, B)) == []

@parameterized([
    ([],     [],           []),
    ([1],    [1],          [(1, 1)]),
    ([1],    [1, 2],       [(1, 1)]),
    ([1, 1], [1, 1],       [(1, 1), (1, 1)]),
    ([1, 1], [1, 1, 2],    [(1, 1), (1, 1)]),
    ([1, 2], [5, 2, 1, 3], [(1, 1), (2, 2)])
])
def test_find_injections_one_possible_with_equality_matching(A, B, injection):
    result = list(find_injections(A, B))
    assert result == [Counter(injection)]


# second char must match
def match(a, b):
    return a[1] == b[1]


@parameterized([
    # no injections possible:
    (["a1", "a1"], ["b1"]),
    (["a1", "a1"], ["b1", "b2"]),
    # one possible:
    (["a1"],       ["b1"],             [("a1", "b1")]),
    (["a1", "a3"], ["b3", "b1"],       [("a1", "b1"), ("a3", "b3")]),
    (["a1", "a1"], ["a2", "b1", "c1"], [("a1", "b1"), ("a1", "c1")]),
    (["a1", "a1"], ["a2", "b1", "c1"], [("a1", "b1"), ("a1", "c1")]),
    (["a1", "b1"], ["c1", "c1", "d2"], [("a1", "c1"), ("b1", "c1")]),
    # multiple possible:
    # all 10 subsets of size 2, of 5 items:
    (["a1", "a1"], ["a1", "b1", "c1", "d1", "e1"],
     [("a1", "a1"), ("a1", "b1")],
     [("a1", "a1"), ("a1", "c1")],
     [("a1", "a1"), ("a1", "d1")],
     [("a1", "a1"), ("a1", "e1")],
     [("a1", "b1"), ("a1", "c1")],
     [("a1", "b1"), ("a1", "d1")],
     [("a1", "b1"), ("a1", "e1")],
     [("a1", "c1"), ("a1", "d1")],
     [("a1", "c1"), ("a1", "e1")],
     [("a1", "d1"), ("a1", "e1")]),
    # all partitions of a set of 4 elements into two partitions of
    # size 2:
    (["a1", "a1", "b1", "b1"], ["w1", "x1", "y1", "z1"],
     [("a1", "w1"), ("a1", "x1"), ("b1", "y1"), ("b1", "z1")],
     [("a1", "w1"), ("b1", "x1"), ("a1", "y1"), ("b1", "z1")],
     [("b1", "w1"), ("a1", "x1"), ("a1", "y1"), ("b1", "z1")],
     [("a1", "w1"), ("b1", "x1"), ("b1", "y1"), ("a1", "z1")],
     [("b1", "w1"), ("a1", "x1"), ("b1", "y1"), ("a1", "z1")],
     [("b1", "w1"), ("b1", "x1"), ("a1", "y1"), ("a1", "z1")]),
    (["a1", "b1"], ["x1", "y1", "z2"],
     [("a1", "x1"), ("b1", "y1")],
     [("a1", "y1"), ("b1", "x1")]),
    (["a1", "a1", "a1", "b2"], ["x1", "x1", "x1", "y1", "y1", "z2", "extra"],
     {("b2", "z2"): 1, ("a1", "x1"): 3},
     {("b2", "z2"): 1, ("a1", "x1"): 2, ("a1", "y1"): 1},
     {("b2", "z2"): 1, ("a1", "x1"): 1, ("a1", "y1"): 2})
])
def test_find_injections_with_nonequality_match(A, B, *injections):
    # We need to compare multisets of multisets, which is a bit fiddly as
    # Counter isn't hashable:
    result = Counter(frozenset(c.items()) for c in find_injections(A, B, match))
    expected = Counter(frozenset(Counter(i).items()) for i in injections)
    assert result == expected


def test_equality_of_expressions():
    assert X == X
    assert X != Y
    assert constant(1) == constant(1)
    assert constant(1) != constant(2)
    assert X + Y == X + Y
    assert X + Y == Y + X
    assert X + Y != X + Z
    # These are a bit lucky, just due to the way - and / are
    # implemented:
    assert X - Y == -Y + X
    assert X / Y == X * (Y ** -1)
    assert log(X) == log(X)
    assert log(X) != exp(X)
    assert log(X) != log(Y)

    assert X * Y == Y * X
    assert X * X.T == X.T * X
    assert X * X.T != X * X
    assert X * X.T == X.T * X

    assert dot(X, Y).T == dot(Y.T, X.T)
    assert dot(X, Y) != dot(Y, X)

    assert sum(X * X.T) == sum(X.T * X)
    assert sum(X * X.T) != trace(X) * trace(X)

    assert trace(dot(X, Y.T)) == sum(Y * X)
    assert dot(dot(X, Y), Z) == dot(X, dot(Y, Z))


def test_match():
    # slightly circular as match is used in einsum.__eq__

    # eye insertion to achieve match:
    assert (X * Y).match(dot(X*Y, Z), Z) == eye(X.shape[1])
    assert X.match(dot(X, Z), Z) == eye(X.shape[1])

    assert (X * y.dimshuffle('x', 0)).match(dot(X, Z), Z) \
        == eye(X.shape[1]) * y.dimshuffle(0, 'x')
    # TODO: it doesn't know that
    # eye() * x.dimshuffle(0, 'x') == eye() * x.dimshuffle('x', 0)
    # (both == diag(x))

    assert (X * Y).match(X * Z, Z) == Y
    assert (X * X).match(X * Z, Z) == X
    assert (X * X).match(Y * Z, Z) is None
    assert (Y * X).match(X * Z, Z) == Y
    assert sum(Y * X).match(sum(X * Z), Z) == Y
    assert dot(X, Y).match(dot(X, Z), Z) == Y

    assert dot(X, X).match(dot(X, Z), Z) == X
    assert dot(X, X.T).match(dot(X, Z), Z) == X.T
    assert dot(X, X.T).match(dot(X.T, Z), Z) is None

    assert dot(X, X*X).match(dot(X, Z), Z) == X*X
    assert dot(X, X*X).match(dot(Z, X*X), Z) == X
    assert dot(X, X*X).match(dot(X*X, Z), Z) is None
    assert dot(X, X*X).match(dot(X, X*Z), Z) == X

    assert trace(dot(X, X)).match(sum(X*Z), Z) == X.T

    assert dot(X, Y).T.match(dot(X, Z), Z) is None
    assert dot(X, Y).T.match(dot(X.T, Z), Z) is None
    assert dot(X, Y).T.match(dot(Z, X.T), Z) == Y.T

    assert dot(X, dot(Y, X)).match(dot(X, Z), Z) == dot(Y, X)

    assert X.match(Z, Z) == X
    assert (X * Y).match(Z, Z) == X*Y
