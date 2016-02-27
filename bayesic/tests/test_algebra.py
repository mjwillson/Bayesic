from ..algebra import *

import numpy.testing as npt




def test_add():
    X = InputExpression('X', 2)
    Y = InputExpression('Y', 2)
    out = Add(X, Y)
    fn = out.compile()
    npt.assert_equal(
        fn(X=[[1,2],[3,4]], Y=[[4,3],[2,1]]),
        [[5,5],[5,5]]
    )


def test_mx_mx_product():
    X = InputExpression('X', 2)
    Y = InputExpression('Y', 2)
    XY = dot(X, Y)
    fn = XY.compile()
    npt.assert_equal(
        fn(X=[[1,2],[3,4]], Y=[[4,3],[2,1]]),
        [[8,5],[20,13]]
    )
