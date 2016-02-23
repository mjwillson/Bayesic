"""Some rudimentary symbolic algebra, which helps with reasoning about
conjugacy. I'd've liked to use just theano, or theano+sympy here, but
sadly we need smarter algebra than theano, and better symbolic tensor
support than sympy.

"""
import theano.tensor as T
from collections import Counter

class Expression(object):
    @property
    def input_types(self):
        """Dict of name to (dtype, ndim)"""
        raise NotImplementedError

    # also must have properties ndim, dtype

    def apply_to_inputs_dict(self, inputs):
        """Like apply, but will accept a dict with other inputs besides ours
        in it and pick out only the relevant ones.

        """
        input_types = self.input_types()
        our_inputs = {name: var for name, var in inputs.items() if name in input_types}
        return self.apply(**our_inputs)

    def apply(self, **inputs):
        """Given theano variables for the inputs, return a theano variable for
        the output.

        """
        raise NotImplementedError

    def terms(self):
        """self must be equivalent to Sum(self.terms)"""
        return [self]


class InputExpression(Expression):
    def __init__(self, name, ndim, dtype='float32'):
        self.name = name
        self.ndim = ndim
        self.dtype = dtype

    @property
    def input_types(self):
        return {self.name: (self.dtype, self.ndim)}

    def apply(self, **inputs):
        return inputs[self.name]

    def __str__(self):
        return self.name


class CompositeExpression(Expression):
    """An expression given in terms of some other parent expressions"""

    def __init__(self, parents):
        self.parents = parents

    @property
    def input_types(self):
        result = {}
        for expr in self.parents:
            for name, type_ in expr.input_types().items():
                if result.get(name, type_) != type_:
                    raise TypeError("same input %s occurs with different types %s, %s" % (type_, result[name]))
                result[name] = type
        return result

    def apply(self, **inputs):
        parent_vars = [parent.apply_to_inputs_dict(inputs) for parent in self.parents]
        return self._apply_to_parents(*parent_vars)

    def _apply_to_parents(self, *parent_vars):
        raise NotImplementedError

    def __str__(self):
        return "%s(%s)" % (type(self).__name__, ', '.join(self.parents))



class Add(CompositeExpression):
    def __init__(self, terms):
        # associativity means we can avoid sums within sums
        flattened_terms = [t for term in terms for t in term.terms()]

        self.ndim = flattened_terms[0].ndim
        self.dtype = flattened_terms[0].dtype
        if not all(t.ndim == self.ndim and t.dtype == self.dtype for t in flattened_terms):
            raise ValueError("Add requires same ndim, dtype on all terms")

        super(Add, self).__init__(flattened_terms)

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


class Einsum(CompositeExpression):
    """This is a general form for various kinds of products between
    tensors, and other multilinear functions of tensors, similar to
    numpy.einsum.

    It can implement:

    dot, tensor_dot, batched_dot, outer, transpose / dimshuffle,
    diagonal, trace, sum along some axes, elemwise product, and
    various things inbetween.

    f(X, Y, Z)_{a,b,c} = sum_{i,j} X_{a,i} Y_{i,j,b} Z_{c, j}
    """

    def __init__(self, factors, indices_for_factors):
        """`factors` -- one or more arguments

        `sum_indices` -- a list of lists of indices, where each index
        is a pair `(factor_number, axis)` representing a particular
        axis of one of the factors. Each list corresponds to a
        summation where all the specified axes are indexed by the
        summation index.

        `output_indices` -- a list of list of indices, like the above.
        The nth list of indices determines the axis which are indexed
        by the nth output index.

        Each axis of each factor must occur once and only once in the
        above lists.
        """
        self.factors = factors
        self.indices_for_factors = indices_for_factors
        if not all(f.ndim == len(indices) for f, indices in zip(factors, indices_for_factors)):
            raise ValueError("The indices for each factor must have same length as factor.ndim")

        output_nums = set(
            num for indices in indices_for_factors
            for type_, num in indices
            if type_ == 'output'
        )
        self.sum_nums = sorted(
            num for _, indices in vars_and_indices
            for type_, num in indices
            if type_ == 'sum'
        )

        if not all(num in output_nums for num in range(len(output_nums))):
            raise ValueError("some output index numbers are missing")
        self.ndim = len(output_nums)
        self.dtype = factors[0].dtype
        if not all(t.dtype == self.dtype for t in flattened_terms):
            raise ValueError("Add requires same ndim, dtype on all terms")

        super(Einsum, self).__init__(factors)

    @property
    def output_type(self):
        return ('float32', self.num_outputs)

    def factors(self):
        return self.parents

    def _eliminate_duplicate_indices(var, indices):
        while True:
            dupe_info = find_duplicate(indices)
            if dupe_info is None:
                return var, indices
            axis1, axis2, dupe = dupe_info
            # Bug with T.diagonal when axes are 0, 1 but ndim > 2
            var = T.Diagonal(0, axis1, axis2)(var)
            indices = tuple([i for n, i in enumerate(indices) if n != axis1 and n != axis2] + [dupe])

    def _apply_to_parents(self, *factor_vars):
        vars_and_indices = [
            self._eliminate_duplicate_indices(var, indices)
            for var, indices in zip(factor_vars, self.indices_for_factors)
        ]

        def axis_of(index, indices, fallback):
            try:
                return indices.index(index)
            except IndexError:
                return 'x'

        def shuffle_to_axes(factor_indices):
            return [axis_of(('output', num), factor_indices) for num in range(self.ndim)] + \
                [axis_of(('sum', num), factor_indices) for num in sum_nums]

        aligned_vars = [
            var.dimshuffle(*shuffle_to_axes(factor_indices))
            for var, factor_indices in vars_and_indices
        ]
        sum_axes_in_output = list(range(len(output_nums), len(output_nums) + len(sum_nums)))
        return T.mul(*aligned_vars).sum(axis=sum_axes_in_output)

    def __str__(self):
        def letter_for_index(type, num):
            if type == 'output':
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
            "%s_%s" % (factor, ''.join(letter_for_index(*i) for i in indices))
            for factor, indices in self.indices_for_factors
        )
        sum_letters = ''.join(letter_for_index('sum', n) for n in self.sum_nums)
        output_letters = ''.join(letter_for_index('output', i) for n in range(self.ndim))
        return 'Einsum(output_%s = sum_%s %s)' % (output_letters, sum_letters, product)