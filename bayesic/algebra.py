"""Some rudimentary symbolic algebra, which helps with reasoning about
conjugacy. I'd've liked to use just theano, or theano+sympy here, but
sadly we need smarter algebra than theano, and better symbolic tensor
support than sympy.

"""
import theano
import theano.tensor as T
from collections import Counter

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

    # also must have properties ndim, dtype

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
        return f

    def _apply_to_parents(self, *parent_vars):
        raise NotImplementedError

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ', '.join(repr(x) for x in self.parents))

    def terms(self):
        """self must be equivalent to Sum(self.terms)"""
        return [self]

    def as_einsum(self):
        indices = [('out', i) for i in range(self.ndim)]
        return Einsum([(self, indices)], self.ndim)


class InputExpression(Expression):
    def __init__(self, name, ndim, dtype='float32'):
        self.name = name
        self.ndim = ndim
        self.dtype = dtype
        super(InputExpression, self).__init__([])

    @property
    def input_types(self):
        return {self.name: (self.dtype, self.ndim)}

    def apply(self, inputs):
        return inputs[self.name]

    def __repr__(self):
        return self.name



class add(Expression):
    def __init__(self, *terms):
        # associativity means we can avoid sums within sums
        flattened_terms = [t for term in terms for t in term.terms()]

        self.ndim = flattened_terms[0].ndim
        self.dtype = flattened_terms[0].dtype
        if not all(t.ndim == self.ndim and t.dtype == self.dtype for t in flattened_terms):
            raise ValueError("Add requires same ndim, dtype on all terms")

        super(add, self).__init__(flattened_terms)

    def terms(self):
        return self.parents

    def _apply_to_parents(self, *term_vars):
        return T.add(*term_vars)


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


class Einsum(Expression):
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

        Einsum([(X, (o2, s0, o1), (Y, (s0, s1, o0, o1)])

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

        # If the factors are themselves Einsums, we swallow them up,
        # incorporating their factors into ours. This means first
        # building a mapping of the sum indices from all the separate
        # Einsum factors, as well as our own sum indices, into a
        # single combined namespace / range:
        sums = 0
        index_mapping = {}
        for factor, indices in factors_and_indices:
            if isinstance(factor, Einsum):
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

        # Now we can map the factors of each parent Einsum factor, up
        # to a factor of ourself.
        self.factors_and_indices = []
        for parent, parent_in_self in factors_and_indices:
            if isinstance(parent, Einsum):
                parent_factors_and_indices = parent.factors_and_indices
            else:
                # Treat parent as an identity einsum if it's not
                # already on Einsum:
                parent_factors_and_indices = [(parent, [('out', i) for i in range(parent.ndim)])]

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

        self.dtype = factors_and_indices[0][0].dtype

        # Could relax this across the board, it'd just mean having to
        # replicate the knowledge about which output dtypes you get
        # from which combinations of inputs:
        if not all(f.dtype == self.dtype for f, _ in factors_and_indices):
            raise ValueError("Einsum requires same dtype on all factors")

        super(Einsum, self).__init__([f for f, _ in self.factors_and_indices])

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

        product = ' '.join(
            "%r_%s" % (factor, ''.join(letter_for_index(*i) for i in indices))
            for factor, indices in self.factors_and_indices
        )
        sum_letters = ''.join(letter_for_index('sum', n) for n in range(self.sums))
        output_letters = ''.join(letter_for_index('out', n) for n in range(self.ndim))
        return 'Einsum(out_%s = sum_%s %s)' % (output_letters, sum_letters, product)


def dot(X, Y):
    """Inner / dot product of two tensors. Sums over the last axis of X
    and the first of Y."""
    X_indices = [('out', i) for i in range(X.ndim-1)] + [('sum', 0)]
    Y_indices = [('sum', 0)] + [('out', i) for i in range(X.ndim - 1, X.ndim + Y.ndim - 2)]
    return Einsum(((X, X_indices), (Y, Y_indices)), X.ndim + Y.ndim - 2)

def mul(*args):
    """Element-wise / hadamard product of some tensors."""
    ndim = args[0].ndim
    if not all(a.ndim == ndim for a in args):
        raise ValueError("all ndim's must match for elemwise mul")

    args_and_indices = [
        (arg, [('out', i) for i in range(arg.ndim)])
        for arg in args
    ]
    return Einsum(args_and_indices, ndim)

def outer(X, Y):
    """Outer product of two tensors, a.k.a. tensor product.
    Unlike theano's this generalises to all tensor shapes.

    Kronecker product is a reshaped version of this applied to matrix
    args.

    """
    X_indices = [('out', i) for i in range(X.ndim)]
    Y_indices = [('out', i) for i in range(X.ndim, X.ndim + Y.ndim)]
    return Einsum([(X, X_indices), (Y, Y_indices)], X.ndim + Y.ndim)

def sum(X, axis=None):
    """Sum entries along all axes, or the given axis (or axes) only if specified"""
    if isinstance(axis, int): axis = [axis]
    if axis is None: axis = range(X.ndim)
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
    return Einsum([(X, indices)], out_index)

def trace(X):
    # TODO could specify axes
    return Einsum([(X, [('sum', 0), ('sum', 0)])], 0)

def diagonal(X):
    # TODO could specify axes
    return Einsum([(X, [('out', 0), ('out', 0)])], 1)

def transpose(X):
    return dimshuffle(X, *reversed(range(X.ndim)))

def dimshuffle(X, *axes):
    """Like theano's dimshuffle. As there, 'x' can be used to indicate a
    broadcastable axis.

    """
    indices = [None] * X.ndim
    for output_no, axis in enumerate(axes):
        if axis != 'x':
            if indices[axis] is not None:
                raise ValueError("dimshuffle: same input axis can't occur twice")
            indices[axis] = ('out', output_no)
    if any(i is None for i in indices):
        raise ValueError("dimshuffle: can't drop an axis")

    return Einsum([(X, indices)], len(axes))

# TODO: tensor_dot, batched_dot
