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
        self.parents = tuple(parents)

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

    def _theano_inputs_and_expression(self):
        input_vars = {
            name: T.TensorType(dtype, [False]*ndim)(name)
            for name, (dtype, ndim) in self.input_types.items()
        }
        expression = self.apply(input_vars)
        return input_vars, expression

    def compile(self):
        input_vars, expression = self._theano_inputs_and_expression()
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
        return repr(self)

    def terms(self):
        """self should be equivalent to sum(self.terms)"""
        return [self]

    def _equality_by(self):
        """Should return a hashable value which equality is based on for
        instances of this type (or override __eq__ and __hash__
        yourself)

        """
        return self.parents

    def __eq__(self, other):
        """Equality of expressions at (or slightly above) the syntax level.

        It does knows about some simple algebraic equivalences, and is
        relatively smart for einsums, but don't count on it doing any
        other more complicated algebraic simplifications in order to
        prove equivalence.

        """
        return isinstance(other, self.__class__) \
            and self._equality_by() == other._equality_by()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self._equality_by())

    @property
    def shape(self):
        return tuple(shape(self, n) for n in range(self.ndim))

    @property
    def size(self):
        return mul(*self.shape)


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

    def _equality_by(self):
        return self.name


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

    def _equality_by(self):
        return self.value


class shape(Expression):
    ndim = 0

    def __init__(self, expression, axis):
        super().__init__([expression])
        self.axis = axis

    def _apply_to_parents(self, expression):
        return expression.shape[self.axis]

    def _equality_by(self):
        return (self.parents[0], self.axis)

    def __repr__(self):
        return "%s.shape[%d]" % (self.parents[0].bracketed_repr(), self.axis)


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
        raise ValueError(
            "Dimension mismatch, was %d, should be %d. If you want broadcasting "
            "you need to do it explicitly via dimshuffle" % (X.ndim, ndim))


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

    def _equality_by(self):
        return (self._apply_to_parents, self.parents)


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

    def bracketed_repr(self):
        return '(%r)' % self

    def _equality_by(self):
        """Making this a set means equality knows about commutativity of addition"""
        return frozenset(self.parents)


class eye(Expression):
    """A square identity matrix"""

    ndim = 2

    @with_wrapped_literals
    def __init__(self, *shapes):
        """shapes should be one or more symbolic or literal scalars, all
        guaranteed to be equal at runtime.

        (Why bother allowing redundant equal copies of the same
        argument? Because we may not know statically which shapes are
        equal, except by dint of them occurring together here; knowing
        all the static shapes equal to ours helps us test equality
        statically with other eye expressions.)

        """
        if len(shapes) == 0:
            raise ValueError("need at least one shape for eye")
        super().__init__(shapes)

    def _apply_to_parents(self, shape, *_):
        return T.eye(shape)

    def __eq__(self, other):
        """Any two eye expressions whose respective set of shape expressions
        overlap, are considered equal.

        This isn't technically an equivalence relation, really we want
        its transitive closure, but we don't have the global
        information to get that. Some kind of clever type inference
        with shape type parameters and shape equality constraints
        could do better, but this is Python not Haskell :)

        Alternatively we could:

        * make all eye's equal without regard for shape, and require
        that you don't try to compare things unless you know their
        shape to be equal.

        * make equality a syntax level thing depending on the
          potentially arbitrary choice of which single symbolic shape
          to pass to it.

        """
        return isinstance(other, self.__class__) and \
            len(set(self.parents) & set(other.parents)) > 0

    def __hash__(self):
        """To be compatible with __eq__ this has to be a not-very-unique hash.
        Still I wouldn't expect loads of these in the same dict/set,
        so should be ok.

        """
        return hash(self.__class__)



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

def equivalence_classes(equal_pairs):
    classes = {}
    for a, b in equal_pairs:
        class_ = classes.get(a, frozenset([a])) | classes.get(b, frozenset([b]))
        for x in class_: classes[x] = class_
    return frozenset(classes.values())


def einsum(factors_and_indices, ndim=None):
    """This is a general form for various kinds of products between
    tensors, and other multilinear functions of tensors, similar to
    numpy.einsum.

    It can implement dot, tensor_dot, batched_dot, outer, transpose /
    dimshuffle, diagonal, trace, sum along some axes, elemwise
    product, and various things inbetween.

    Params:

    `factors_and_indices` -- pairs of (factor, indices) where indices
    is a tuple containing an index for each axis of the factor.

    Indices are either ('sum', n) or ('out', n).

    A sum index will be summed over; an out index corresponds to
    an axis of the output tensor. E.g. let s0, s1 be sum
    indices, and o0, o1, o2 output indices. Then

    einsum([(X, (o2, s0, o1), (Y, (s0, s1, o0, o1)])

    corresponds to a tensor T whose entries are:

    T_{o0, o1, o2} = sum_{s0, s1} X_{o2, s0, o1} Y_{s0, s1, o0, o1}

    `ndim` is the number of 'out' indices. If not specified it'll be
    set to max output index + 1. Note that not every output index in
    range(0, ndim) needs to occur -- if an output index doesn't occur,
    it'll correspond to a broadcastable axis in the output tensor (see
    docs on theano broadcasting)

    """
    return Einsum(factors_and_indices, ndim)._canonicalize()


class Einsum(Expression):
    """See docs on `einsum`"""

    @classmethod
    def _wrap_if_not_einsum(cls, expr):
        """Wraps with an identity-equivalent Einsum if not one already."""
        if isinstance(expr, cls):
            return expr
        else:
            return cls([(expr, tuple(('out', i) for i in range(expr.ndim)))], expr.ndim)

    def __init__(self, factors_and_indices, ndim=None):
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
        self.factors_and_indices = tuple(factors_and_indices)
        super(Einsum, self).__init__([f for f, _ in factors_and_indices])

    @property
    def out_indices(self):
        return [('out', i) for i in range(self.ndim)]

    @property
    def sum_indices(self):
        return sorted(set(
            i for factor, indices in self.factors_and_indices
            for i in indices
            if i[0] == 'sum'
        ))

    def _canonicalize(self):
        """This is done automatically by einsum.__new__"""
        return self \
            ._incorporate_child_einsum_factors() \
            ._collapse_eye_factors() \
            ._passthrough_if_identity()

    def _incorporate_child_einsum_factors(self):
        """If the factors are themselves einsums, we swallow them up,
        incorporating their factors into ours. This means first
        building a mapping of the sum indices from all the separate
        einsum factors, as well as our own sum indices, into a single
        combined namespace / range.

        """
        sums = 0
        index_mapping = {}
        for factor, indices in self.factors_and_indices:
            if isinstance(factor, Einsum):
                for sum_index in factor.sum_indices:
                    index_mapping[(factor, sum_index)] = ('sum', sums)
                    sums += 1

        for factors, indices in self.factors_and_indices:
            for index in indices:
                if index[0] == 'sum' and index not in index_mapping:
                    index_mapping[index] = ('sum', sums)
                    sums += 1

        def map_idx(index, parent=None):
            # sum indices will get mapped; out indices passed through.
            return index_mapping.get(
                (parent, index) if parent is not None else index,
                index
            )

        # Now we can map the factors of each parent einsum factor, up
        # to a factor of ourself.
        result_factors_and_indices = []
        for parent, parent_in_self in self.factors_and_indices:
            if isinstance(parent, Einsum):
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
                        factor_in_self.append(map_idx(parent_in_self[i]))
                    elif type_ == 'sum':
                        # sum index with respect to this particular
                        # parent einsum. We look up what top-level sum
                        # index to map it to.
                        factor_in_self.append(map_idx(('sum', i), parent))
                result_factors_and_indices.append((factor, tuple(factor_in_self)))

        return Einsum(result_factors_and_indices, self.ndim)

    def _collapse_eye_factors(self):
        """Remove eye / identity-matrix / delta factors, when one of their
        indices is summed over.

        """
        equal_index_pairs = [(i, i) for i in self.sum_indices]
        def filter_eye_with_sum_index(factor, indices):
            if isinstance(factor, eye) and (indices[0][0] == 'sum' or indices[1][0] == 'sum'):
                equal_index_pairs.append(indices)
                return False
            return True

        self.factors_and_indices = [
            fi for fi in self.factors_and_indices if filter_eye_with_sum_index(*fi)]

        eye_index_mapping = {}
        for class_ in equivalence_classes(equal_index_pairs):
            # this will pick an output index if present, otherwise the
            # lowest sum index. ('out' < 'sum' -- convenient)
            representative = min(class_)
            for index in class_: eye_index_mapping[index] = representative

        # Now renumber to fill gaps
        sum_nos = sorted(
            index_no for type_, index_no in eye_index_mapping.values()
            if type_ == 'sum')

        renumber_mapping = {('sum', no): ('sum', i) for i, no in enumerate(sum_nos)}

        def collapse_eye_indices(factor, indices):
            def collapse_index(i):
                i = eye_index_mapping.get(i, i)
                return renumber_mapping.get(i, i)
            return factor, tuple(collapse_index(i) for i in indices)

        result_factors_and_indices = tuple(
            collapse_eye_indices(*fi) for fi in self.factors_and_indices)
        return Einsum(result_factors_and_indices, self.ndim)

    def _passthrough_if_identity(self):
        if len(self.factors_and_indices) == 1:
            factor, indices = self.factors_and_indices[0]
            if (self.ndim == factor.ndim and
                all(index == ('out', j) for index, j in zip(indices, range(factor.ndim)))):
                return factor
        return self

    @property
    def output_type(self):
        return ('float32', self.num_outputs)

    def factors(self):
        return self.parents

    def _rewrite_repeated_index_on_factor_as_diagonal(self):
        def rewrite_factor(factor, indices):
            while True:
                dupe_info = find_duplicate(indices)
                if dupe_info is None:
                    return factor, indices
                axis1, axis2, dupe = dupe_info
                factor = _diagonal(factor, axis1, axis2)
                indices = tuple([i for n, i in enumerate(indices)
                                 if n != axis1 and n != axis2] + [dupe])
        result_factors_and_indices = tuple(
            rewrite_factor(f, i) for f, i in self.factors_and_indices)
        return Einsum(result_factors_and_indices, self.ndim)

    def _rewrite_as_special_case_ops(self):
        """So we've gone to all the length of unifying various kinds of einsum
        based expressions, but to actually execute them we need to
        convert them back to some combination of special-purpose ops:
        diagonal, sum, mul and dimshuffle are technically sufficient,
        although we also use tensordot (which generalises dot) where
        possible to make things fast.

        Special internal expression classes (_sum, _mul etc) are to
        represent these special-purpose ops as a step along the way to
        generating a theano expression.

        Summations are done in the order specified -- this means that
        the associativity/bracketing of a bunch of matrix products is
        preserved, so you have control over this (it can affect
        compute time). If we were doing this at runtime we could look
        at the shapes of the factors and figure out the actual best
        ordering of operations then, but without knowing the shapes,
        some static heuristics together with preserving the requested
        ordering of summations will have to do.

        """
        return self \
            ._rewrite_repeated_index_on_factor_as_diagonal() \
            ._rewrite_as_sum_of_mul()

    def _rewrite_as_sum_of_mul(self):
        """A naive approach to implementing an einsum: dimshuffle all tensors
        so they line up, with one axis per (sum or output) index and
        so that they broadcast over any indices that don't appear.
        Then just multiply them, and then sum over the sum axes.

        This can be very slow, since it creates a big (potentially
        high-dimensional) intermediate result tensor.

        There must be no repeated indices on any factor before calling
        this (see _rewrite_repeated_index_on_factor_as_diagonal)

        TODO: we can do better than this by spotting sums that are
        reducible to matrix-matrix, matrix-vector products. For now
        just worrying about the algebra though.

        """
        def axis_of(index, indices):
            try:
                return indices.index(index)
            except ValueError:
                return 'x'

        def dimshuffle_to_axes(factor, indices):
            axes = [axis_of(i, indices) for i in self.out_indices + self.sum_indices]
            if axes == list(range(factor.ndim)):
                return factor
            else:
                return _dimshuffle(factor, *axes)

        aligned_factors = [
            dimshuffle_to_axes(factor, indices)
            for factor, indices in self.factors_and_indices
        ]
        sum_axes_in_output = list(range(self.ndim, self.ndim + len(self.sum_indices)))
        if len(aligned_factors) == 0:
            mul_result = dimshuffle(constant(1), *(['x'] * self.ndim))
        elif len(aligned_factors) == 1:
            mul_result = aligned_factors[0]
        else:
            mul_result = _mul(*aligned_factors)
        if sum_axes_in_output:
            return _sum(mul_result, *sum_axes_in_output)
        else:
            return mul_result

    def apply(self, inputs):
        return self._rewrite_as_special_case_ops().apply(inputs)

    def __repr__(self):
        def letter_for_index(type, num):
            if type == 'out':
                try:
                    return 'uvwxyz'[num]
                except IndexError:
                    return 'o%d' % num
            elif type == 'sum':
                num = self.sum_indices.index((type, num))
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
            indexed_factor_repr(factor, indices) for factor, indices in self.factors_and_indices) \
            or '1'

        if self.sum_indices:
            sum_letters = ''.join(letter_for_index(*i) for i in self.sum_indices)
            summed_product = 'sum_%s %s' % (sum_letters, product)
        else:
            summed_product = product

        if self.ndim > 0:
            output_letters = ''.join(letter_for_index(*i) for i in self.out_indices)
            return 'einsum(out_%s = %s)' % (output_letters, summed_product)
        else:
            return 'einsum(%s)' % summed_product

    @staticmethod
    def _factor_axes_for_indices(indices_for_factors):
        index_to_axes = defaultdict(set)
        for factor_no, indices in enumerate(indices_for_factors):
            for axis_no, index in enumerate(indices):
                index_to_axes[index].add((factor_no, axis_no))
        return {index: frozenset(axes) for index, axes in index_to_axes.items()}

    def match(self, template, slot):
        template = Einsum._wrap_if_not_einsum(template)

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
        nonslots = [(f, i) for f, i in template.factors_and_indices if f is not slot]
        factor_injections = find_injections(nonslots, self_factors_and_indices,
                                            match=lambda a, b: a[0] == b[0])
        for factor_injection in factor_injections:
            remaining_factors_and_indices = self_factors_and_indices - Counter(
                {fi: c for (_, fi), c in factor_injection.items()})

            if len(factor_injection) == 0:
                factors, template_indices, self_indices = [], [], []
            else:
                factors, template_indices, self_indices = zip(*(
                    (factor, template_indices_for_factor, self_indices_for_factor)
                    for (factor, template_indices_for_factor),
                        (factor, self_indices_for_factor) in factor_injection.elements())
                )

            template_indices_and_axes = self._factor_axes_for_indices(template_indices).items()
            self_indices_and_axes = self._factor_axes_for_indices(self_indices).items()

            def index_match(self_index_and_axes, template_index_and_axes):
                (template_index_type, template_index_no), template_axes \
                    = template_index_and_axes
                (self_index_type, self_index_no), self_axes = self_index_and_axes

                # must occur in the same positions (axes of factors):
                if template_axes != self_axes: return False
                if self_index_type == 'sum':
                    # sum index must map to sum index (but the index
                    # number doesn't have to be the same)
                    return template_index_type == 'sum'
                elif self_index_type == 'out':
                    # out index can map to a sum index or an out
                    # index, but if it's an out index then the output
                    # number must match.
                    #
                    # Where an out index maps to a sum index in the
                    # template, to obtain a match we can turn the out
                    # index into a sum index by adding an extra
                    # delta_ij (identity matrix) factor and summing
                    # over it, where i is the sum index and j is the
                    # output index in question. See later for this.
                    return (
                        template_index_type != 'out' or
                        template_index_no == self_index_no
                    )

            index_bijection = find_bijection(self_indices_and_axes, template_indices_and_axes,
                                             match=index_match)
            if index_bijection is not None:
                # We found a valid match of the non-slot factors of
                # the template, and their indices. Now we just need to
                # translate the remaining terms into something
                # subtitutable for the slot.
                to_template_index = {
                    self_index: template_index
                    for (self_index, _), (template_index, _) in index_bijection.elements()
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

                    elif index_type == 'out':
                        # Some output indices might exist only in the
                        # remaining factors but not the matches for
                        # the nonslots, so won't be in this bijection,
                        # these we just let map to the same output
                        # index.
                        #
                        # Others will map either to the same out
                        # index, or a sum index in the template:
                        template_index = to_template_index.get(self_index, self_index)
                        slot_axis = template_index_to_slot_axis.get(template_index)
                        if slot_axis is None: return
                        return ('out', slot_axis)

                result_factors_and_indices = [
                    (factor, [to_result_index(i) for i in indices])
                    for factor, indices in remaining_factors_and_indices.elements()
                ]
                for (self_index, axes), (template_index, _) in index_bijection.elements():
                    if self_index[0] == 'out' and template_index[0] == 'sum':
                        # we need to convert this output index into a
                        # sum index by adding a delta_ij / identity
                        # matrix factor making the two indices equal
                        # (and summing over the sum index)

                        # Find which indices in the slot the sum
                        # index, and the output index (which should be
                        # equivalent to it) map to, if any:
                        sum_index_as_slot_axis = template_index_to_slot_axis.get(template_index)
                        out_index_as_slot_axis = template_index_to_slot_axis.get(self_index)
                        sum_index_as_result_index = ('out', sum_index_as_slot_axis) \
                                                     if sum_index_as_slot_axis is not None else None
                        out_index_as_result_index = ('out', out_index_as_slot_axis) \
                                                     if out_index_as_slot_axis is not None else None
                        # we sort them so that if the desired output
                        # slot eye is symmetric so it doesn't matter
                        # which way round the indices go, but einsum
                        # doesn't know this, so we sort the two
                        # indices -- if they're in order (and without
                        # gaps) einsum will just pass through the eye
                        # matrix without needing to wrap it.
                        eye_result_indices = sorted(
                            [sum_index_as_result_index, out_index_as_result_index])
                        # Get its shape from all the factor axes it
                        # occurs in:
                        matching_shapes = [
                            factors[factor_no].shape[axis_no] for factor_no, axis_no in axes]
                        # Add the eye factor forcing the two indices equal
                        result_factors_and_indices.append(
                            (eye(*matching_shapes), eye_result_indices))

                # If to_result_index returned None for any indices, we
                # can't fit the remaining items in the slot, so give
                # up on this match. Otherwise, we're done!
                if not any(i is None for _, indices in result_factors_and_indices for i in indices):
                    return einsum(result_factors_and_indices, ndim=slot.ndim)

    def __eq__(self, other):
        """We test equality by searching for an isomorphism using match. Sadly
        it doesn't seem possible to find a uniquely-identifying
        representative for the equivalence class that isn't sensitive
        to numbering of the sum indices, nor to the ordering of the
        factors. The case when a single factor occurs twice makes this
        hard, in particular cases like sum_ij X_ii X_jj != sum_ij X_ij
        X_ji == sum_ij X_ji X_ij.

        """
        # shortcut for a common case, as this is otherwise
        # expensive-ish for an equality test:
        if self is other: return True
        if not isinstance(other, self.__class__): return False
        slot = var('__slot__', ndim=0)
        match = self.match(other * slot, slot)
        return match is not None and len(match.factors_and_indices) == 0

    def __hash__(self):
        """As per comments on __eq__, finding a canonical equivalence class
        representative to hash doesn't seem easy, so instead we hash
        something which doesn't always uniquely identify self
        (although it usually will). This is OK provided different
        hashes implies nonequality, which is still the case.

        """
        # we represent each sum index as a multiset of (factor, axis)
        # for how many times it occurs on a particular axis of a
        # particular factor (which recall may occur multiple times):
        sum_index_to_axes = defaultdict(Counter)
        for factor, indices in self.factors_and_indices:
            for axis_no, index in enumerate(indices):
                if index[0] == 'sum':
                    sum_index_to_axes[index][(factor, axis_no)] += 1
        sum_index_to_axes = {index: frozenset(axes.items())
                             for index, axes in sum_index_to_axes.items()}

        factor_axes_for_indices = self._factor_axes_for_indices(
            indices for factor, indices in self.factors_and_indices)

        def index_to_insensitive_representation(index):
            index_type, index_no = index
            if index_type == 'out':
                return index
            elif index_type == 'sum':
                return ('sum', sum_index_to_axes[index])

        factors_and_insensitive_indices = Counter(
            (factor, tuple(index_to_insensitive_representation(i) for i in indices))
            for factor, indices in self.factors_and_indices
        )
        return hash(frozenset(factors_and_insensitive_indices.items()))


def match(expression, template, slot):
    """Try to pattern-match an expression against a template einsum
    expression which contains a var `slot` as exactly one of its
    factors.

    Returns an expression which, when substituted for `slot` in the
    template, is equal to self -- if this is possible, otherwise
    None.

    This is a general way of pulling out some factor(s) out of an
    einsum, and of testing if it has a particular factor(s),
    noting that it's not just the identity of the factor that
    matters, but the way it's multiplied/combined with other
    factors (as specified by the way the `slot` is combined with
    the other terms in the template). Useful for identifying and
    collecting like terms in a su

    Note this can create identity (eye) factors to pull out, even if
    they weren't present in the original expression, where this
    enables a match. E.g. matching `A` against `dot(A, X)` works,
    giving `X = eye(A.shape[1])`.

    In future this may expand to pattern-match broader classes of
    expression.

    """
    return Einsum._wrap_if_not_einsum(expression).match(template, slot)


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
    A_size = _builtin_sum(A.values())
    B_size = _builtin_sum(B.values())
    if A_size != B_size:
        return []
    else:
        return find_injections(A, B, match)


def find_injection(A, B, match=lambda a, b: a == b):
    return next(iter(find_injections(A, B, match)), None)


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


_builtin_sum = sum


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


# Some special-case ops used internally when generating an
# implementation expression from an einsum. No argument checking is
# done for these, since not for public consumption.

class _sum(Expression):
    def __init__(self, X, *axes):
        self.axes = axes
        self.ndim = X.ndim - len(axes)
        super().__init__([X])

    def _apply_to_parents(self, X):
        return X.sum(axis=self.axes)


class _mul(Expression):
    def __init__(self, *factors):
        self.ndim = factors[0].ndim
        super().__init__(factors)

    def _apply_to_parents(self, *factors):
        if len(factors) == 1:
            return factors[0]
        else:
            return T.mul(*factors)


class _dimshuffle(Expression):
    def __init__(self, X, *axes):
        self.axes = axes
        self.ndim = X.ndim
        super().__init__([X])

    def _apply_to_parents(self, X):
        return X.dimshuffle(*self.axes)


class _tensordot(Expression):
    def __init__(self, X, Y, X_axes, Y_axes):
        self.X_axes = X_axes
        self.Y_axes = Y_axes
        self.ndim = X.ndim + Y.ndim - len(X_axes) - len(Y_axes)
        super().__init__([X, Y])

    def _apply_to_parents(self, X, Y):
        # TODO maybe call dot directly if only one sum axis
        return T.tensordot(X, Y, (self.X_axes, self.Y_axes))


class _diagonal(Expression):
    def __init__(self, X, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2
        self.ndim = X.ndim - 1
        super().__init__([X])

    def _apply_to_parents(self, X):
        # Bug with T.diagonal when axes are 0, 1 but ndim > 2
        return T.Diagonal(0, self.axis1, self.axis2)(X)


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
