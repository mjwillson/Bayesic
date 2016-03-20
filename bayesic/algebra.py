"""Some rudimentary symbolic algebra, which helps with reasoning about
conjugacy. I'd've liked to use just theano, or theano+sympy here, but
sadly we need smarter algebra than theano, and better symbolic tensor
support than sympy.

"""
import numpy as np
import theano
import theano.tensor as T

import itertools as it
from collections import Counter, defaultdict

class Expression(object):
    def __init__(self, parents):
        self.parents = parents

    @property
    def input_types(self):
        """Dict of name to (dtype, ndim)"""
        result = {}
        for expr in self.parents:
            for name, type_ in expr.input_types.items():
                if result.get(name, type_) != type_:
                    raise TypeError("same input %s occurs with different types %s, %s" % (type_, result[name]))
                result[name] = type_
        return result

    # also must have property ndim

    def apply(self, inputs):
        """Given theano variables for the inputs, return a theano variable for
        the output.

        """
        parent_vars = [parent.apply(inputs) for parent in self.parents]
        return self._apply_to_parents(*parent_vars)

    def compile(self):
        input_vars = {
            name: T.TensorType(dtype, [False]*ndim)(name)
            for name, (dtype, ndim) in self.input_types.items()
        }
        expression = self.apply(input_vars)
        inputs = list(input_vars.values())
        input_names = list(input_vars.keys())
        fn = theano.function(inputs, expression)
        def f(**inputs):
            return fn(*(inputs[name] for name in input_names))
        f.theano_fn = fn
        return f

    def _apply_to_parents(self, *parent_vars):
        raise NotImplementedError

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ', '.join(repr(x) for x in self.parents))

    def bracketed_repr(self):
        return '(%r)' % self

    def terms(self):
        """self must be equivalent to Sum(self.terms)"""
        return [self]



class var(Expression):
    def __init__(self, name, ndim, dtype='float32'):
        self.name = name
        self.ndim = ndim
        self.dtype = dtype
        super(var, self).__init__([])

    @property
    def input_types(self):
        return {self.name: (self.dtype, self.ndim)}

    def apply(self, inputs):
        return inputs[self.name]

    def __repr__(self):
        return self.name

    bracketed_repr = __repr__


class constant(Expression):
    def __init__(self, value):
        self.value = value
        self._constant = T.constant(value)
        self.ndim = self._constant.ndim
        self.dtype = self._constant.dtype
        super(constant, self).__init__([])

    def apply(self, inputs):
        return self._constant

    def __repr__(self):
        return repr(self.value)

    bracketed_repr = __repr__


def wrap_if_literal(x):
    if np.isscalar(x) or isinstance(x, np.ndarray):
        return constant(x)
    elif isinstance(x, Expression):
        return x
    else:
        raise ValueError("must be a scalar, numpy array or Expression")


def with_wrapped_literals(fn):
    def wrapped_fn(*args):
        return fn(*(wrap_if_literal(x) for x in args))
    return wrapped_fn


def autobroadcast_or_match(X, ndim):
    """Ensure `X` has `ndim`.

    A scalar get auto-broadcasted, but otherwise `ndim` must match.

    """
    if X.ndim == ndim:
        return X
    elif X.ndim == 0:
        return dimshuffle(X, *(['x'] * ndim))
    else:
        raise ValueError("Dimension mismatch. If you want broadcasting you need to do it explicitly via dimshuffle")


class elemwise(Expression):
    """Wraps an element-wise theano op (or op-like-function)"""
    def __init__(self, theano_op, *args, name=None):
        args = [wrap_if_literal(x) for x in args]
        self.ndim = max(arg.ndim for arg in args)
        args = [autobroadcast_or_match(arg, self.ndim) for arg in args]
        self._apply_to_parents = theano_op
        self.name = name or theano_op.scalar_op.name
        super(elemwise, self).__init__(args)

    def __repr__(self):
        return "%s(%s)" % (self.name, ', '.join(repr(x) for x in self.parents))


class add(elemwise):
    def __init__(self, *terms):
        # associativity means we can avoid sums within sums
        flattened_terms = [
            t for term in terms for t in wrap_if_literal(term).terms()]
        super(add, self).__init__(T.add, *flattened_terms)

    def terms(self):
        return self.parents

    def _apply_to_parents(self, *term_vars):
        return T.add(*term_vars)

    def __repr__(self):
        return ' + '.join(repr(x) for x in self.parents)


def find_duplicate(values):
    """Returns `(index1, index2, value)`, the first duplicate value
    occurring in `values` and the two first indices at which it
    occurs, or None if no duplicates.

    """
    seen = {}
    for i, v in enumerate(values):
        if v in seen:
            return i, seen[v], v
        seen[v] = i


class einsum(Expression):
    """This is a general form for various kinds of products between
    tensors, and other multilinear functions of tensors, similar to
    numpy.einsum.

    It can implement:

    dot, tensor_dot, batched_dot, outer, transpose / dimshuffle,
    diagonal, trace, sum along some axes, elemwise product, and
    various things inbetween.
    """

    def __init__(self, factors_and_indices, ndim=None):
        """`factors_and_indices` -- pairs of (factor, indices)
        where indices contains an index for each axis of the factor.

        Indices are either ('sum', n) or ('out', n).

        A sum index will be summed over; an out index corresponds to
        an axis of the output tensor. E.g. let s0, s1 be sum
        indices, and o0, o1, o2 output indices. Then

        einsum([(X, (o2, s0, o1), (Y, (s0, s1, o0, o1)])

        corresponds to a tensor T whose entries are:

        T_{o0, o1, o2} = sum_{s0, s1} X_{o2, s0, o1} Y_{s0, s1, o0, o1}

        `ndim` is the number of 'out' indices. If not specified it'll
        be set to max index + 1. Note that not every output index in
        range(0, ndim) needs to occur -- if an output index doesn't
        occur, it'll correspond to a broadcastable axis in the output
        tensor (see docs on theano broadcasting)

        """
        if not all(f.ndim == len(indices) for f, indices in factors_and_indices):
            raise ValueError("The indices for each factor must have same length as factor.ndim")

        output_nums = [
            num for factors, indices in factors_and_indices
            for type_, num in indices
            if type_ == 'out'
        ]
        if ndim is None:
            ndim = max(output_nums) + 1
        if not all(num >= 0 and num < ndim for num in output_nums):
            raise ValueError("some output indices are out of range")
        self.ndim = ndim

        # If the factors are themselves einsums, we swallow them up,
        # incorporating their factors into ours. This means first
        # building a mapping of the sum indices from all the separate
        # einsum factors, as well as our own sum indices, into a
        # single combined namespace / range:
        sums = 0
        index_mapping = {}
        for factor, indices in factors_and_indices:
            if isinstance(factor, einsum):
                for i in range(factor.sums):
                    index_mapping[(factor, ('sum', i))] = ('sum', sums)
                    sums += 1

        for factors, indices in factors_and_indices:
            for type_, i in indices:
                if type_ == 'sum' and (self, ('sum', i)) not in index_mapping:
                    index_mapping[(self, ('sum', i))] = ('sum', sums)
                    sums += 1

        self.sums = sums
        def map_idx(parent, i):
            # sum indices will get mapped; out indices passed through.
            return index_mapping.get((parent, i), i)

        # Now we can map the factors of each parent einsum factor, up
        # to a factor of ourself.
        self.factors_and_indices = []
        for parent, parent_in_self in factors_and_indices:
            if isinstance(parent, einsum):
                parent_factors_and_indices = parent.factors_and_indices
            else:
                # Treat parent as an identity einsum if it's not
                # already on einsum:
                parent_factors_and_indices = [
                    (wrap_if_literal(parent), [('out', i) for i in range(parent.ndim)])]

            for factor, factor_in_parent in parent_factors_and_indices:
                factor_in_self = []
                for type_, i in factor_in_parent:
                    if type_ == 'out':
                        # out index with respect to its parent -- we
                        # then look up what index that axis of the
                        # parent expression, has in the ourself (the
                        # top-level expression). In case that index is
                        # a sum index, we need to map it as our
                        # top-level sum indices may have been shifted
                        # up by all the sum indices taken from the
                        # einsums beneath us. If it's an out index for
                        # the top-level expression then it gets left
                        # as such.
                        factor_in_self.append(map_idx(self, parent_in_self[i]))
                    elif type_ == 'sum':
                        # sum index with respect to this particular
                        # parent einsum. We look up what top-level sum
                        # index to map it to.
                        factor_in_self.append(map_idx(parent, ('sum', i)))
                self.factors_and_indices.append((factor, factor_in_self))

        super(einsum, self).__init__([f for f, _ in self.factors_and_indices])

    @property
    def output_type(self):
        return ('float32', self.num_outputs)

    def factors(self):
        return self.parents

    def _eliminate_duplicate_indices(self, var, indices):
        while True:
            dupe_info = find_duplicate(indices)
            if dupe_info is None:
                return var, indices
            axis1, axis2, dupe = dupe_info
            # Bug with T.diagonal when axes are 0, 1 but ndim > 2
            var = T.Diagonal(0, axis1, axis2)(var)
            indices = tuple([i for n, i in enumerate(indices) if n != axis1 and n != axis2] + [dupe])

    def _apply_to_parents(self, *factor_vars):
        # First eliminate any duplicate indices into the same input
        # tensor, via extracting diagonals. This will ensure each
        # index appears at most once in each tensor, which will help
        # later:
        vars_and_indices = [
            self._eliminate_duplicate_indices(var, indices)
            for var, (factor, indices) in zip(factor_vars, self.factors_and_indices)
        ]

        # Now a simple approach: dimshuffle all tensors so they line
        # up, with one axis per (sum or output) index and so that they
        # broadcast over any indices that don't appear. Then just
        # multiply them, and then sum over the sum axes.
        #
        # This can be very slow, since it creates a big (potentially
        # high-dimensional) intermediate result tensor.
        #
        # TODO: we can do better than this by spotting sums that are
        # reducible to matrix-matrix, matrix-vector products. For now
        # just worrying about the algebra though.

        def axis_of(index, indices):
            try:
                return indices.index(index)
            except ValueError:
                return 'x'

        def shuffle_to_axes(factor_indices):
            return [axis_of(('out', num), factor_indices) for num in range(self.ndim)] + \
                [axis_of(('sum', num), factor_indices) for num in range(self.sums)]

        aligned_vars = [
            var.dimshuffle(*shuffle_to_axes(factor_indices))
            for var, factor_indices in vars_and_indices
        ]
        sum_axes_in_output = list(range(self.ndim, self.ndim + self.sums))
        return T.mul(*aligned_vars).sum(axis=sum_axes_in_output)

    def as_einsum(self):
        return self

    def __repr__(self):
        def letter_for_index(type, num):
            if type == 'out':
                try:
                    return 'uvwxyz'[num]
                except IndexError:
                    return 'o%d' % num
            elif type == 'sum':
                try:
                    return 'ijklmn'[num]
                except IndexError:
                    return 's%d' % num

        def indexed_factor_repr(factor, indices):
            if len(indices) == 0:
                return factor.bracketed_repr()
            else:
                return "%s_%s" % (factor.bracketed_repr(),
                                  ''.join(letter_for_index(*i) for i in indices))

        product = ' '.join(
            indexed_factor_repr(factor, indices) for factor, indices in self.factors_and_indices)

        if self.sums > 0:
            sum_letters = ''.join(letter_for_index('sum', n) for n in range(self.sums))
            summed_product = 'sum_%s %s' % (sum_letters, product)
        else:
            summed_product = product

        if self.ndim > 0:
            output_letters = ''.join(letter_for_index('out', n) for n in range(self.ndim))
            return 'einsum(out_%s = %s)' % (output_letters, summed_product)
        else:
            return 'einsum(%s)' % summed_product

    def match(self, template, slot):
        """Try to pattern-match this einsum expression against a template
        einsum expression which contains a var `slot` as exactly one
        of its factors.

        Returns an einsum which, when substituted for `slot` in the
        template, is equal to self -- if this is possible, otherwise
        None.

        This is a general way of pulling out some factor(s) out of an
        einsum, and of testing if it has a particular factor(s),
        noting that it's not just the identity of the factor that
        matters, but the way it's multiplied/combined with other
        factors (as specified by the way the `slot` is combined with
        the other terms in the template). Useful for identifying and
        collecting like terms in a sum.

        """
        try:
            slot_indices = next(
                indices for factor, indices in template.factors_and_indices if factor is slot)
        except StopIteration:
            raise ValueError("template must contain slot as a factor")

        template_index_to_slot_axis = {}
        for axis, index in enumerate(slot_indices):
            if index in template_index_to_slot_axis:
                # Note we could support this using diag
                # expressions, but it's not a very useful thing to
                # match and also makes life more fiddly, so not
                # bothering for now.
                raise ValueError("Same index used on multiple slot axes is not currently supported")
            template_index_to_slot_axis[index] = axis

        self_factors_and_indices = Counter(self.factors_and_indices)
        nonslots = [fi for fi in template.factors_and_indices if factor is not slot]
        factor_injections = find_injections(nonslots, self_factors_and_indices,
                                            match=lambda a, b: a[0] == b[0])
        for factor_injection in factor_injections:
            remaining_factors_and_indices = self_factors_and_indices - Counter(
                {fi: c for (_, fi), c in factor_injection.items()})

            factors, template_indices, self_indices = zip(*(
                (factor, template_indices_for_factor, self_indices_for_factor)
                for (factor, template_indices_for_factor),
                    (factor, self_indices_for_factor) in factor_injection.elements())
            )

            def factor_axes_for_indices(indices_for_factors):
                index_to_axes = defaultdict(set)
                for factor_no, indices in enumerate(indices_for_factors):
                    for axis_no, index in enumerate(indices):
                        index_to_axes[index].add((factor_no, axis_no))
                return [(index, frozenset(axes)) for index, axes in index_to_axes.items()]

            template_indices_and_axes = factor_axes_for_indices(template_indices)
            self_indices_and_axes = factor_axes_for_indices(self_indices)

            def index_match(template_index_and_axes, self_index_and_axes):
                (template_index_type, template_index_no), template_axes \
                    = template_index_and_axes
                (self_index_type, self_index_no), self_axes = self_index_and_axes

                return (
                    # must occur in the same positions (axes of factors):
                    template_axes == self_axes and
                    # must be of same type (sum or out)
                    template_index_type == self_index_type and
                    # number must match for output index
                    (template_index_no == self_index_no if template_index_type == 'out' else True)
                )

            index_bijection = find_bijection(template_indices_and_axes, self_indices_and_axes,
                                             match=index_match)
            if index_bijection is not None:
                # We found a valid match of the non-slot factors of
                # the template, and their indices. Now we just need to
                # translate the remaining terms into something
                # subtitutable for the slot.
                to_template_index = {
                    self_index: template_index
                    for (template_index, _), (self_index, _) in index_bijection.elements()
                }

                def to_result_index(self_index):
                    index_type, index_no = self_index
                    if index_type == 'sum':
                        if self_index in to_template_index:
                            # A sum index which bridges the slot with
                            # the other non-slot factors whose indices
                            # we have mapped to template indices.

                            # Because of this, we can look at the
                            # template to find which axis of the slot
                            # we need to output to, to get this sum
                            # index.
                            template_index = to_template_index[self_index]
                            slot_axis = template_index_to_slot_axis.get(template_index)
                            # The index might not be used in the slot
                            # at all -- in which case we can't output
                            # to it / can't fit this remaining term in
                            # the slot:
                            if slot_axis is None: return
                            # Otherwise we're good, we can output to
                            # the slot axis which corresponds to this
                            # sum index:
                            return ('out', slot_axis)

                        else:
                            # this is a sum index which occurs only in
                            # the remaining terms, not in the (matches
                            # for the) non-slot factors. We re-use the
                            # same sum index number as a sum index in
                            # our result. (Re-using ensures the
                            # original ordering of sum indices is
                            # preserved)
                            return self_index

                    else:
                        # the index bijection preserves output
                        # indices, so we don't need to do a lookup for
                        # these. Also some output indices might exist
                        # only in the remaining factors but not the
                        # matches for the nonslots, so won't be in
                        # this bijection.
                        template_index = self_index
                        slot_axis = template_index_to_slot_axis.get(template_index)
                        # The output index might not be used in the
                        # slot at all -- in which case we can't output
                        # to it / can't fit this remaining term in the
                        # slot:
                        if slot_axis is None: return
                        # Otherwise we're good, we can output to
                        # the slot axis which corresponds to this
                        # (outer) output index:
                        return ('out', slot_axis)

                result_factors_and_indices = [
                    (factor, [to_result_index(i) for i in indices])
                    for factor, indices in remaining_factors_and_indices.elements()
                ]
                # If to_result_index returned None for any indices, we
                # can't fit the remaining items in the slot, so give
                # up on this match. Otherwise, we're done!
                if not any(i is None for _, indices in result_factor_and_indices for i in indices):
                    return einsum(result_factors_and_indices, ndim=slot.ndim)


def submultisets_of_size(A, n):
    """Enumerate all sub-multisets of a given multiset, with a given size.

    A should be a Counter.

    It might help to view this algorithm as a generalization of
    "generate all subsets of size n of a set", whose inductive step is
    f(S, n) = f(S \ {s}, n) + f(S \ {s}, n-1)

    """
    if not A:
        if n == 0: yield Counter([])
        return
    A_remaining = A.copy()
    a, a_count = A_remaining.popitem()

    for count_to_include in range(0, min(a_count, n)+1):
        a_s = Counter({a: count_to_include})
        for S in submultisets_of_size(A_remaining, n - count_to_include):
            yield a_s + S


def find_bijection(A, B, match=lambda a, b: a == b):
    return next(iter(find_bijections(A, B, match)), None)


def find_bijections(A, B, match=lambda a, b: a == b):
    A = Counter(A)
    B = Counter(B)
    A_size = sum(A.values())
    B_size = sum(B.values())
    if A_size != B_size:
        return []
    else:
        return find_injection(A, B, match)


def find_injections(A, B, match=lambda a, b: a == b):
    """Find all injections (i.e. one-to-one mappings) f from multiset A to
    multiset B, satisfying match(a, f(a)) for all a.

    A and B can be given as lists, or as Counters/Counter-like dicts
    with counts as values.

    Injections are yielded as Counters of pairs (a, b), such that:

    Counter(b for a, b in f) is a sub-multiset of Counter(B)

    i.e. this is an injection of multisets -- each b in B can be
    mapped to at most once if it occurs once in B, at most twice if it
    occurs twice etc.

    all(match(a, b) for a, b in f) -- the match property requested

    Counter(A) == Counter(a for a, b in f) -- i.e. has a corresponding
    occurrence in B for every occurrence in A

    """
    def _find_injections(A, B):
        if not A:
            yield Counter([])
            return
        A_remaining = A.copy()
        a, a_count = A_remaining.popitem()

        # Enumerate all the ways to allocate a's count to matching b's:
        all_matching_bs = Counter({b: count for b, count in B.items() if match(a, b)})
        for matching_bs in submultisets_of_size(all_matching_bs, a_count):
            ab_s = Counter({(a, b): n for b, n in matching_bs.items()})
            # Then recursively match up all the remaining items. This
            # may blow the stack for large input. Input not expected
            # to be large, but if that changes could rewrite
            B_remaining = B - matching_bs
            for inj in _find_injections(A_remaining, B_remaining):
                yield ab_s + inj

    return _find_injections(Counter(A), Counter(B))


# einsum-based ops:

@with_wrapped_literals
def dot(X, Y):

    """Inner / dot product of two tensors. Sums over the last axis of X
    and the first of Y."""
    X_indices = [('out', i) for i in range(X.ndim-1)] + [('sum', 0)]
    Y_indices = [('sum', 0)] + [('out', i) for i in range(X.ndim - 1, X.ndim + Y.ndim - 2)]
    return einsum(((X, X_indices), (Y, Y_indices)), X.ndim + Y.ndim - 2)


@with_wrapped_literals
def mul(*args):
    """Element-wise / hadamard product of some tensors."""
    ndim = max(arg.ndim for arg in args)
    args = [autobroadcast_or_match(arg, ndim) for arg in args]

    args_and_indices = [
        (arg, [('out', i) for i in range(arg.ndim)])
        for arg in args
    ]
    return einsum(args_and_indices, ndim)


@with_wrapped_literals
def outer(X, Y):
    """Outer product of two tensors, a.k.a. tensor product.
    Unlike theano's this generalises to all tensor shapes.

    Kronecker product is a reshaped version of this applied to matrix
    args.

    """
    X_indices = [('out', i) for i in range(X.ndim)]
    Y_indices = [('out', i) for i in range(X.ndim, X.ndim + Y.ndim)]
    return einsum([(X, X_indices), (Y, Y_indices)], X.ndim + Y.ndim)


def sum(X, axis=None):
    """Sum entries along all axes, or the given axis (or axes) only if specified"""
    if isinstance(axis, int): axis = [axis]
    if axis is None: axis = range(X.ndim)
    X = wrap_if_literal(X)
    indices = []
    sum_index = 0
    out_index = 0
    for i in range(X.ndim):
        if i in axis:
            indices.append(('sum', sum_index))
            sum_index += 1
        else:
            indices.append(('out', out_index))
            out_index += 1
    return einsum([(X, indices)], out_index)


@with_wrapped_literals
def trace(X):
    # TODO could specify axes
    return einsum([(X, [('sum', 0), ('sum', 0)])], 0)


@with_wrapped_literals
def diagonal(X):
    # TODO could specify axes
    return einsum([(X, [('out', 0), ('out', 0)])], 1)


@with_wrapped_literals
def transpose(X):
    return dimshuffle(X, *reversed(range(X.ndim)))


def dimshuffle(X, *axes):
    """Like theano's dimshuffle. As there, 'x' can be used to indicate a
    broadcastable axis.

    """
    X = wrap_if_literal(X)
    indices = [None] * X.ndim
    for output_no, axis in enumerate(axes):
        if axis != 'x':
            if indices[axis] is not None:
                raise ValueError("dimshuffle: same input axis can't occur twice")
            indices[axis] = ('out', output_no)
    if any(i is None for i in indices):
        raise ValueError("dimshuffle: can't drop an axis")

    return einsum([(X, indices)], len(axes))

# TODO: tensor_dot, batched_dot

# Elemwise ops

@with_wrapped_literals
def div(X, Y):
    """Element-wise division"""
    # doing it this way allows it to participate in einsum (via mul)
    return mul(X, Y ** -1)


@with_wrapped_literals
def neg(X):
    return -1 * X


@with_wrapped_literals
def sub(self, other):
    return add(self, -other)


def log(X):
    return elemwise(T.log, X)


def exp(X):
    return elemwise(T.exp, X)


def pow(X, Y):
    return elemwise(T.pow, X, Y)


def abs_(X):
    return elemwise(T.abs_, X)


# Add operator overloading:

def _swap(fn):
    def swapped(x, y):
        return fn(y, x)
    return swapped

# add is a class, have to wrap to use as bindable method
def _add(*args):
    return add(*args)
Expression.__add__ = _add
Expression.__radd__ = _swap(add)
Expression.__sub__ = sub
Expression.__rsub__ = _swap(sub)
Expression.__mul__ = mul
Expression.__rmul__ = _swap(mul)
Expression.__truediv__ = div
Expression.__rtruediv__ = _swap(div)
Expression.__pow__ = pow
Expression.__rpow__ = _swap(pow)
Expression.__matmul__ = dot
Expression.__rmatmul__ = _swap(dot)
Expression.__neg__ = neg
Expression.__abs__ = abs_
Expression.T = property(transpose)
Expression.dimshuffle = dimshuffle
Expression.sum = sum
Expression.dot = dot
