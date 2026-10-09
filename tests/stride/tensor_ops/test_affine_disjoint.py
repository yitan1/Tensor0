"""Producer output independence survives forward AD, not reverse accumulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import SU2Irrep, hom, space
from tensor0.structure import get_degeneracystructure
from tensor0._stride._jax import accumulation_p
from tensor0._stride._layout import AffineRecord
from tensor0._stride._tensor_ops import _strided_affine_transform
from tests.stride.support.availability import native_available


RECORDS = tuple(AffineRecord((4,), (1,), 0, (2,), offset) for offset in (0, 1))


def independent(source, factor):
    return accumulation_p.bind(source, factor, records=RECORDS,
        coefficient_records=(0,), output_size=8, dtype=np.dtype('float32'),
        outputs_disjoint=True)


def reference(source, factor):
    return jnp.stack((source * factor, source), axis=-1).reshape(8)


def accumulation_equations(function, *arguments):
    return [equation for equation in jax.make_jaxpr(function)(*arguments).jaxpr.eqns
            if equation.primitive is accumulation_p]


def test_affine_producer_proof_and_repeated_destination_fallback():
    factor_space = space(SU2Irrep, {0: 2, 1: 3})
    layout = get_degeneracystructure(hom((factor_space, factor_space),
                                       (factor_space, factor_space)))
    blocks = layout.subblockstructure
    first, second = next((i, j) for i in range(len(blocks))
                         for j in range(i + 1, len(blocks))
                         if tuple(blocks[i].sizes) == tuple(blocks[j].sizes))
    source = jnp.ones(layout.total_dim, dtype=jnp.float32)

    def operation(data, coefficient, destinations):
        return _strided_affine_transform(data, source_subblocks=blocks,
            destination_subblocks=blocks,
            entries=tuple((first, destination, coefficient) for destination in destinations),
            permutation=(), output_size=layout.total_dim, result_dtype=data.dtype,
            shape_error='unexpected shape')

    for destinations, expected in (((first, second), True), ((first, first), False)):
        equations = accumulation_equations(
            lambda data, coefficient: operation(data, coefficient, destinations),
            source, jnp.float32(2))
        assert len(equations) == 1
        assert equations[0].params['outputs_disjoint'] is expected


def test_forward_proof_is_retained_but_reverse_repeated_reads_drop_it():
    source, factor = jnp.arange(4, dtype=jnp.float32), jnp.float32(1.25)
    forward = accumulation_equations(
        lambda data, tangent: jax.jvp(lambda value: independent(value, factor),
                                     (data,), (tangent,)), source, source)
    assert len(forward) == 2
    assert all(equation.params['outputs_disjoint'] for equation in forward)
    coefficient_forward = accumulation_equations(
        lambda coefficient: jax.jvp(lambda alpha: independent(source, alpha),
                                   (coefficient,), (jnp.float32(.5),)), factor)
    assert len(coefficient_forward) == 2
    assert all(equation.params['outputs_disjoint'] for equation in coefficient_forward)
    reverse = accumulation_equations(
        lambda cotangent: jax.vjp(lambda value: independent(value, factor), source)[1](cotangent),
        jnp.ones(8, dtype=jnp.float32))
    reverse = [equation for equation in reverse if equation.params['output_size'] == 4]
    assert reverse
    assert all(not equation.params.get('outputs_disjoint', False) for equation in reverse)
    batched = accumulation_equations(jax.vmap(independent), jnp.stack((source, source)),
                                    jnp.asarray([0., 2.], dtype=jnp.float32))
    assert len(batched) == 1 and batched[0].params['outputs_disjoint']


@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_disjoint_forward_reverse_coefficients_and_second_order():
    source = jnp.asarray([1., -2., 3., .25], dtype=jnp.float32)
    cotangent = jnp.arange(8, dtype=jnp.float32) / 8
    compiled = jax.jit(independent)
    for factor in (0., 1., -1.25):
        factor = jnp.float32(factor)
        np.testing.assert_array_equal(compiled(source, factor), reference(source, factor))
        actual = jax.jit(lambda data, coefficient: (
            jax.jvp(independent, (data, coefficient), (jnp.ones_like(data), jnp.float32(.5))),
            jax.vjp(independent, data, coefficient)[1](cotangent)))(source, factor)
        expected = (jax.jvp(reference, (source, factor), (jnp.ones_like(source), jnp.float32(.5))),
                    jax.vjp(reference, source, factor)[1](cotangent))
        for value, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(value, wanted, rtol=2e-6, atol=2e-6)
    hessian = lambda operation: jax.jacfwd(jax.grad(
        lambda coefficient: jnp.sum(operation(source, coefficient) ** 2)))
    np.testing.assert_allclose(jax.jit(hessian(independent))(jnp.float32(0)),
                               hessian(reference)(jnp.float32(0)), rtol=2e-6, atol=2e-6)
    text = compiled.lower(source, jnp.float32(2)).as_text()
    assert 'outputs_disjoint = 1' in text
    generic = jax.jit(lambda value: accumulation_p.bind(value, records=(RECORDS[0],) * 2,
        output_size=8, dtype=value.dtype))
    np.testing.assert_array_equal(generic(source), reference(source, jnp.float32(2)).at[1::2].set(0))
    assert 'outputs_disjoint = 0' in generic.lower(source).as_text()
