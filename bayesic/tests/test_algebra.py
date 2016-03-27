from ..algebra import *
# We use these internals to check that under-the-hood einsum rewriting
# is happening in an expected and efficient way:
from ..algebra import _sum, _mul, _dimshuffle, _tensordot, _diagonal

import numpy.testing as npt
import numpy as np
import numpy.random as npr
import theano
import theano.tensor as T
from theano.printing import debugprint as Tprint

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
W = var('W', 2)
X_ = randn(5, 5)
Y_ = randn(5, 5)

# Some vectors
x = var('x', 1)
y = var('y', 1)
x_ = randn(5)
y_ = randn(5)

# 3D tensor
S = var('S', 3)
S_ = randn(3, 5, 7)

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
    fn = dimshuffle(S, 2, 0, 1).compile()
    npt.assert_allclose(fn(S=S_), np.transpose(S_, (2, 0, 1)))


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
    fn = sum(S).compile()
    npt.assert_allclose(fn(S=S_), S_.sum(), rtol=1e-5)

    fn = sum(S, axis=0).compile()
    npt.assert_allclose(fn(S=S_), S_.sum(axis=0), rtol=1e-5)

    # Check works as a method too
    fn = S.sum(axis=(0, 2)).compile()
    npt.assert_allclose(fn(S=S_), S_.sum(axis=(0, 2)), rtol=1e-5)


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
def _match(a, b):
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
    result = Counter(frozenset(c.items()) for c in find_injections(A, B, _match))
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
    # slightly circular as match is used in einsum.__eq__,
    # but still useful to test

    assert match(X * Y, X * Z, Z) == Y
    assert match(X * X, X * Z, Z) == X
    assert match(X * X, Y * Z, Z) is None
    assert match(Y * X, X * Z, Z) == Y
    assert match(sum(Y * X), sum(X * Z), Z) == Y
    assert match(dot(X, Y), dot(X, Z), Z) == Y

    assert match(dot(X, X), dot(X, Z), Z) == X
    assert match(dot(X, X.T), dot(X, Z), Z) == X.T
    assert match(dot(X, X.T), dot(X.T, Z), Z) is None

    assert match(dot(X, X*X), dot(X, Z), Z) == X*X
    assert match(dot(X, X*X), dot(Z, X*X), Z) == X
    assert match(dot(X, X*X), dot(X*X, Z), Z) is None
    assert match(dot(X, X*X), dot(X, X*Z), Z) == X

    assert match(trace(dot(X, X)), sum(X*Z), Z) == X.T

    assert match(dot(X, Y).T, dot(X, Z), Z) is None
    assert match(dot(X, Y).T, dot(X.T, Z), Z) is None
    assert match(dot(X, Y).T, dot(Z, X.T), Z) == Y.T

    assert match(dot(X, dot(Y, X)), dot(X, Z), Z) == dot(Y, X)

    assert match(X, Z, Z) == X
    assert match(X * Y, Z, Z) == X*Y


def test_identity_elimination():
    assert dot(X, eye(X.shape[1])) == X
    assert dot(eye(X.shape[0]), X) == X
    assert dot(Y, dot(eye(X.shape[0]), X)) == dot(Y, X)


def test_identity_insertion_to_achieve_match():
    assert match(X * Y, dot(X*Y, Z), Z) == eye(X.shape[1])
    assert match(X, dot(X, Z), Z) == eye(X.shape[1])
    assert match(X * y.dimshuffle('x', 0), dot(X, Z), Z) \
        == eye(X.shape[1]) * y.dimshuffle(0, 'x')
    # TODO: it doesn't know that
    # eye() * x.dimshuffle(0, 'x') == eye() * x.dimshuffle('x', 0)
    # (both == diag(x))


def assert_implemented_as(expr, impl_expr):
    rewritten = expr._rewrite_as_special_case_ops()
    assert rewritten == impl_expr, "%r != %r" % (rewritten, impl_expr)


# Warning: these einsum-rewriting tests are slightly too specific in
# what they test for -- in practise there are a few equivalent outputs
# that could be acceptable, usually differing only by presence of
# extra dimshuffles, flipping lhs / rhs of a tensordot etc (so e.g.
# based on identities like XY = (Y^T X^T)^T). That said we do have
# some logic that tries to make pretty / conventional choices in these
# cases that preserve the original expression order when it's not too
# hard to do this, so perhaps fair to test for this.


def test_basic_einsum_rewriting():
    assert_implemented_as(
        diagonal(X),
        _diagonal(X, 0, 1)
    )
    assert_implemented_as(
        dot(X, Y),
        _tensordot(X, Y, [1], [0])
    )
    assert_implemented_as(
        sum(X, 1),
        _sum(X, 1)
    )
    assert_implemented_as(
        mul(X, Y),
        _mul(X, Y)
    )
    assert_implemented_as(
        dimshuffle(X, 1, 0),
        _dimshuffle(X, 1, 0)
    )


def test_einsum_rewriting_can_generate_nested_tensordots_preserving_specified_dot_bracketing():
    assert_implemented_as(
        dot(X, dot(Y, Z)),
        _tensordot(X, _tensordot(Y, Z, [1], [0]), [1], [0])
    )
    assert_implemented_as(
        dot(dot(X, Y), Z),
        _tensordot(_tensordot(X, Y, [1], [0]), Z, [1], [0])
    )
    assert_implemented_as(
        dot(dot(X, Y), dot(Z, W)),
        _tensordot(_tensordot(X, Y, [1], [0]),
                   _tensordot(Z, W, [1], [0]),
                   [1], [0])
    )
    assert_implemented_as(
        dot(dot(X, dot(Y, Z)), W),
        _tensordot(
            _tensordot(
                X,
                _tensordot(Y, Z, [1], [0]),
                [1], [0]
            ),
            W,
            [1], [0]
        )
    )

    # Just a reminder that we know these are algebraically equivalent
    # -- still in practise the order we do them in can matter for
    # compute cost, so we don't want to throw that information away
    # when doing this fancy everything-goes-via-einsums stuff. The
    # information is kept in the ordering of the sum indices.
    assert dot(X, dot(Y, Z)) == dot(dot(X, Y), Z)
    assert dot(dot(X, Y), dot(Z, W)) == dot(dot(X, dot(Y, Z)), W)


def test_einsum_rewriting_no_summation():
    assert_implemented_as(
        X * Y.T * x.dimshuffle(0, 'x'),
        _mul(X, _dimshuffle(Y, 1, 0), _dimshuffle(x, 0, 'x'))
    )


def test_einsum_rewriting_summation_affecting_one_term_only():
    # can't generate a dot product in these cases
    assert_implemented_as(
        X.sum(1) * y,
        _mul(_sum(X, 1), y)
    )


def test_einsum_rewriting_as_tensordot():
    # hadamard product of matrices, implemented as a tensor dot
    # product:
    assert_implemented_as(
        trace(dot(X.T, Y)),
        _tensordot(X, Y, [0, 1], [0, 1])
    )

    # mul and row-wise sum as a batched vector-vector dot product:
    assert_implemented_as(
        (X * Y.T).sum(axis=1),
        _tensordot(
            X, Y,
            X_dot_axes=[1], Y_dot_axes=[0],
            X_batch_axes=[0], Y_batch_axes=[1]
        )
    )


def test_einsum_rewriting_is_smart_about_grouping_terms_into_lhs_and_rhs_when_generating_a_tensordot():
    # this could be implemented a few ways:
    # dot(Z, x*y), dot(Z*x, y), dot(Z*y, x)
    # the first is best as it requires the least broadcasting.
    assert_implemented_as(
        dot(Z, x * y),
        _tensordot(Z, _mul(x, y), [1], [0])
    )
    assert_implemented_as(
        dot(Z*x.dimshuffle('x', 0), y),
        _tensordot(Z, _mul(x, y), [1], [0])
    )
    assert_implemented_as(
        dot(Z*y.dimshuffle('x', 0), x),
        _tensordot(Z, _mul(x, y), [1], [0])
    )

    # These two expressions are equivalent. The former is better (just
    # a mx-mx dot product and two matrix elemwise muls, vs going via
    # 3d tensors and doing a bunch of broadcasting). The rewriting
    # process should choose it in both cases when it decides how to
    # split factors between the lhs and the rhs of the _tensordot it
    # generates.
    assert_implemented_as(
        dot(X*Y, Z*W),
        _tensordot(_mul(X, Y), _mul(Z, W), [1], [0])
    )
    assert_implemented_as(
        tensordot(
            X.dimshuffle(0,1,'x') * Z.dimshuffle('x',0,1),
            Y.dimshuffle(0,1,'x') * W.dimshuffle('x',0,1),
            X_sum_axes=[1], Y_sum_axes=[1],
            X_batch_axes=[0, 2], Y_batch_axes=[0, 2]
        ),
        _tensordot(_mul(X, Y), _mul(Z, W), [1], [0])
    )
