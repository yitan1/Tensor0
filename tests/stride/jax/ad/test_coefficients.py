"""JAX AD coefficients contracts."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, dotc, dotu
from tensor0._stride._jax import accumulation_p, reduction_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.coefficients import (
    DISCRETE,
    SHARED_CASES,
    check,
    shared_coefficient_case,
)
from tests.stride.support.oracles.dot import dot_functions
from tests.stride.support.oracles.effective_factor import SINGLE
from tests.stride.support.oracles.reduction import (
    compare_derivatives,
    functions as reduction_ad_functions,
)
from tests.stride.support.oracles.scalar import PARTIAL, assert_components
from tests.stride.support.oracles.update import LAYOUTS, compare, update_functions
from tests.stride.support.samples import values


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


INTEGERS = ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')


COEFFICIENT_PAIRS = (
    ('float16', 'float16'),
    ('bfloat16', 'bfloat16'),
    ('float32', 'complex64'),
    ('complex64', 'float16'),
    ('float64', 'complex128'),
    ('complex128', 'float64'),
)


def assert_close(actual, expected):
    for value, target in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert value.dtype == target.dtype and value.shape == target.shape
        np.testing.assert_allclose(value, target, rtol=0.002, atol=0.002)


# AD uses bool, signed and unsigned source representatives, not every width.
# Keep all coefficient pairs unbatched, then one source per pair across batch
# shapes. Source-type interactions are intentionally sampled, including empties.
COEFFICIENT_CASES = [
    (source, coefficient, result, ())
    for source in ('bool', 'int32', 'uint64') for coefficient, result in COEFFICIENT_PAIRS
] + [
    (source, coefficient, result, batch)
    for source, coefficient, result in (
        ('bool', 'float16', 'float16'),
        ('bool', 'bfloat16', 'bfloat16'),
        ('int32', 'float32', 'complex64'),
        ('int32', 'complex64', 'float16'),
        ('uint64', 'float64', 'complex128'),
        ('uint64', 'complex128', 'float64'),
    )
    for batch in ((2,), (0,), (2, 0))
]


@partial(jax.jit, static_argnums=0)
def integer_coefficient_ad(execute, source, arguments, cotangent):
    # Keep the integer source dynamic, including when differentiating only the
    # coefficients. Do not compile four pullbacks with captured source buffers.
    run = lambda *factors: execute(source, *factors)
    directions = tuple(jnp.ones_like(value) for value in arguments)
    return (jax.jvp(run, arguments, directions),
            jax.vjp(run, *arguments)[1](cotangent),
            jax.linear_transpose(run, *arguments)(cotangent),
            jax.vjp(execute, source, *arguments)[1](cotangent))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('source_dtype,coefficient_dtype,dtype,batch_shape', COEFFICIENT_CASES)
@pytest.mark.parametrize('operation', ['accumulation', 'reduction'])
def test_coefficient_derivatives(source_dtype, coefficient_dtype, dtype, operation, batch_shape):
    source = jnp.broadcast_to(jnp.asarray([0, 1, 2, 3, 1], dtype=source_dtype), (*batch_shape, 5))
    arguments = (jnp.asarray(2 + 1j if coefficient_dtype.startswith('complex') else 2, dtype=coefficient_dtype), jnp.full(batch_shape, 0.5, dtype=dtype))
    execute, oracle = reduction_ad_functions(operation, dtype)
    reference = lambda *factors: oracle(source, *factors)
    directions = tuple(jnp.ones_like(value) for value in arguments)
    expected_forward = jax.jvp(reference, arguments, directions)
    output = expected_forward[0]
    cotangent = jnp.full_like(output, 1 + 1j if jnp.iscomplexobj(output) else 1)
    expected = jax.vjp(reference, *arguments)[1](cotangent)
    forward, reverse, transpose, gradients = integer_coefficient_ad(execute, source, arguments, cotangent)
    assert_close(forward, expected_forward)
    assert_close(reverse, expected)
    assert_close(transpose, expected)
    assert gradients[0].dtype == jax.dtypes.float0 and gradients[0].shape == source.shape
    assert_close(gradients[1:], expected)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation', ['accumulation', 'reduction'])
@pytest.mark.parametrize('factor', [0, 1, 2])
def test_higher_derivatives_and_lowering(operation, factor):
    source = jnp.asarray([1, 2, 3, 1, 2], dtype=jnp.int32)
    run, reference = reduction_ad_functions(operation, 'float32')
    fixed = jnp.float32(0.5)

    def loss(function, value):
        result = function(source, value, fixed)
        return jnp.sum(result * result)
    value = jnp.float32(factor)
    gradient = jax.grad(lambda coefficient: loss(run, coefficient))
    oracle = jax.grad(lambda coefficient: loss(reference, coefficient))
    assert_close(jax.jit(jax.jvp, static_argnums=0)(gradient, (value,), (jnp.float32(1),)), jax.jvp(oracle, (value,), (jnp.float32(1),)))
    linear = lambda coefficient: run(source, coefficient, fixed)
    reverse = jax.jit(jax.vjp(linear, value)[1])
    text = reverse.lower(jnp.ones(6, dtype=jnp.float32)).as_text()
    assert 'tensor0_stride_dot_' in text
    assert 'stablehlo.gather' not in text and 'stablehlo.scatter' not in text


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation', ['accumulation', 'reduction'])
def test_higher_derivatives_vmap(operation):
    source = jnp.asarray([1, 2, 3, 1, 2], dtype=jnp.int32)
    run, reference = reduction_ad_functions(operation, 'float32')
    fixed = jnp.float32(0.5)

    def loss(function, value):
        result = function(source, value, fixed)
        return jnp.sum(result * result)
    gradient = jax.grad(lambda coefficient: loss(run, coefficient))
    oracle = jax.grad(lambda coefficient: loss(reference, coefficient))
    assert_close(jax.jit(jax.vmap(gradient))(jnp.asarray([0.0, 1.0, 2.0])), jax.vmap(oracle)(jnp.asarray([0.0, 1.0, 2.0])))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation', [dotu, dotc])
@pytest.mark.parametrize('source_dtype', INTEGERS)
def test_dot_pullback_with_integer_constant(operation, source_dtype):
    source = jnp.asarray([0, 1, 2], dtype=source_dtype)
    data = jnp.asarray([1, 2, 3], dtype=jnp.float16)
    view = lambda value: StridedView(value, (3,), (1,), 0)
    for constant_left in (False, True):
        run = (lambda value: operation(view(source), view(value))) if constant_left else lambda value: operation(view(value), view(source))
        reverse = jax.vjp(run, data)[1]
        coefficient = jnp.float16(1)
        expected = source.astype(data.dtype)
        value, tangent = jax.jit(lambda cotangent: jax.jvp(reverse, (cotangent,), (cotangent,)))(coefficient)
        assert_close(value[0], expected)
        assert_close(tangent[0], expected)
        actual = jax.jit(jax.linear_transpose(reverse, coefficient))((data,))[0]
        assert_close(actual, jnp.sum(data * expected))


THREE_WAY_AD = [
    ('float16', 'bfloat16', 'float32'), ('bfloat16', 'float16', 'float64'),
    ('float32', 'bfloat16', 'float16'), ('float64', 'float16', 'bfloat16'),
    ('float16', 'complex128', 'float32'), ('float32', 'complex128', 'float16'),
    ('float16', 'float64', 'complex128'), ('complex128', 'float64', 'float16'),
    ('bfloat16', 'complex64', 'float64'), ('float64', 'complex64', 'bfloat16'),
    ('bfloat16', 'float32', 'complex64'), ('complex64', 'float32', 'bfloat16'),
]


# Same-type half/bfloat, widening, narrowing, low-precision coefficients,
# real/complex projection, and the distinct three-way interactions above.
SHARED_AD_TYPES = [
    ('float16', 'float16', 'float16'), ('bfloat16', 'bfloat16', 'bfloat16'),
    ('float16', 'float16', 'float32'), ('float32', 'float32', 'float16'),
    ('bfloat16', 'bfloat16', 'float64'), ('float64', 'float64', 'bfloat16'),
    ('float32', 'float16', 'float32'), ('complex128', 'bfloat16', 'complex128'),
    ('float16', 'complex64', 'complex64'), ('complex64', 'complex64', 'float16'),
] + THREE_WAY_AD


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation,source_dtype,coefficient_dtype,dtype',
                         [case for case in SHARED_CASES
                          if case[1:] in SHARED_AD_TYPES])
def test_shared_coefficient_derivatives(source_dtype, coefficient_dtype, dtype, operation):
    run, reference, arguments = shared_coefficient_case(source_dtype, coefficient_dtype, dtype, operation)
    if operation == 'update':
        compare(run, reference, arguments)
    else:
        compare_derivatives(run, reference, arguments)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
@pytest.mark.parametrize('batch_shape', [(2,), (0,), (2, 0)])
@pytest.mark.parametrize('form', ['scalar', 'one', 'batch'])
def test_update_batch_coefficients(dtype, batch_shape, form):
    shape = {'scalar': (), 'one': (1,), 'batch': batch_shape}[form]
    run, reference = update_functions(LAYOUTS[1])
    arguments = (values((*batch_shape, 6), dtype), values((*batch_shape, 10), dtype), jnp.full(shape, 2, dtype=dtype), jnp.full(shape, 0.5, dtype=dtype))
    compare(run, reference, arguments)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_pullback_lowering_and_vmap(dtype):
    data = values((5,), dtype)
    run, reference = dot_functions(dotu, ((2, 2), (1, 1), 0, (0, -1), 3), dtype)
    reverse = jax.vjp(lambda source: run(source, data), data)[1]
    oracle = jax.vjp(lambda source: reference(source, data), data)[1]
    cotangents = jnp.asarray([0, 1, 2], dtype=dtype)
    actual = jax.jit(jax.vmap(reverse))(cotangents)[0]
    expected = jax.vmap(oracle)(cotangents)[0]
    np.testing.assert_array_equal(actual, expected)
    second_transpose = jax.jit(jax.linear_transpose(reverse, cotangents[0]))
    text = second_transpose.lower((data,)).as_text()
    assert 'tensor0_stride_dot_' in text
    assert 'stablehlo.gather' not in text
    assert 'stablehlo.scatter' not in text


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation', ['accumulation', 'reduction', 'update'])
def test_wide_coefficient_gradient_range(operation):
    source = jnp.full(5, 1e+20, dtype=jnp.float32)
    factor = jnp.float64(1e-30)
    if operation == 'update':
        execute, _ = update_functions((AffineRecord((1,), (1,), 0, (1,), 1),))
        run = lambda coefficient: execute(source, jnp.zeros(3, jnp.float32), coefficient, jnp.int32(0))
        count = 1
    else:
        execute, _ = reduction_ad_functions(operation, 'float32')
        run = lambda coefficient: execute(source, coefficient, jnp.float32(0))
        count = 4
    result, pullback = jax.vjp(run, factor)
    gradient = jax.jit(pullback)(jnp.full_like(result, 1e+20))[0]
    assert gradient.dtype == factor.dtype
    np.testing.assert_allclose(gradient, count * np.float64(jnp.float32(1e+20)) ** 2, rtol=1e-13)


@pytest.mark.usefixtures("enable_x64")
# Dtype dispatch is exhaustive on nonempty inputs; batch/empty behavior uses
# one representative of each discrete family rather than every integer width.
@pytest.mark.parametrize('operation', ['reduction', 'accumulation'])
@pytest.mark.parametrize('dtype,batch_shape',
                         [(dtype, ()) for dtype in DISCRETE]
                         + [(dtype, batch) for dtype in ('bool', 'int32', 'uint64')
                            for batch in ((2,), (0,), (2, 0))])
def test_scaled_discrete_results(operation, dtype, batch_shape):
    records = (AffineRecord((2, 2), (1, 1), 0, (1, 1), 1), AffineRecord((0,), (1,), 3, (1,), 3))
    source = jnp.broadcast_to(jnp.asarray([1, 2, 3], dtype=jnp.float32), (*batch_shape, 3))
    factor = jnp.full(batch_shape, 2, dtype=jnp.float16)

    def run(data, coefficient):
        parameters = dict(records=records, coefficient_records=(0,), output_size=4, dtype=jnp.dtype(dtype))
        if operation == 'reduction':
            return reduction_p.bind(data, coefficient, output_shapes=((2, 1), (1,)), reduction_axes=((False, True), (True,)), **parameters)
        return accumulation_p.bind(data, coefficient, **parameters)
    check(run, (source, factor))


def test_data_derivative_uses_the_composed_factor_without_source_prescaling():
    source = jnp.asarray([30000], dtype=jnp.float16)
    direction = jnp.asarray([65504], dtype=jnp.float16)
    operation = lambda values: update_p.bind(
        values, jnp.zeros_like(values), jnp.float16(2) * jnp.float16(.5),
        jnp.float16(0), records=SINGLE)
    tangent = jax.jit(lambda values, delta: jax.jvp(operation, (values,), (delta,))[1])(source, direction)
    cotangent = jax.jit(lambda values, delta: jax.vjp(operation, values)[1](delta)[0])(source, direction)
    np.testing.assert_array_equal(tangent, direction)
    np.testing.assert_array_equal(cotangent, direction)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.complex64])
def test_independent_effective_factors_batch_and_differentiate(dtype):
    with jax.enable_x64():
        factors = (np.float64(1.0003), np.float32(-.75))
        records = tuple(AffineRecord((2,), (2,), offset, (2,), offset + 1) for offset in range(2))
        base = jnp.ones((3, 6), dtype=dtype)
        source = jnp.asarray([[1., 2., 3., 4.]] * 3, dtype=dtype)
        alpha, beta = jnp.asarray([0., 1., 1.0003]), jnp.asarray([1., 0., -.25])

        def operation(old, values, first, second):
            result = old
            for record, factor in zip(records, factors, strict=True):
                result = update_p.bind(values, result, first * factor, second, records=(record,))
            return result

        def reference(old, values, first, second):
            result = old
            for offset, factor in enumerate(factors):
                selected = (first[:, None] * factor * values[:, offset::2]
                            + second[:, None] * old[:, offset + 1:offset + 4:2])
                result = result.at[:, offset + 1:offset + 4:2].set(selected.astype(dtype))
            return result

        arguments = (base, source, alpha, beta)
        tangents = tuple(jnp.ones_like(value) for value in arguments)
        expected = reference(*arguments)
        np.testing.assert_allclose(jax.jit(operation)(*arguments), expected, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(jax.jit(jax.vmap(operation))(*arguments), expected, rtol=1e-3, atol=1e-6)
        actual_jvp = jax.jit(lambda *values: jax.jvp(operation, values, tangents)[1])(*arguments)
        expected_jvp = jax.jvp(reference, arguments, tangents)[1]
        np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=1e-3, atol=1e-6)
        cotangent = jnp.ones_like(base)
        actual_vjp = jax.jit(lambda *values: jax.vjp(operation, *values)[1](cotangent))(*arguments)
        expected_vjp = jax.vjp(reference, *arguments)[1](cotangent)
        for actual_part, expected_part in zip(actual_vjp, expected_vjp, strict=True):
            np.testing.assert_allclose(actual_part, expected_part, rtol=1e-3, atol=1e-6)
        loss = lambda coefficient: jnp.real(operation(base, source, coefficient, beta)).sum()
        expected_loss = lambda coefficient: jnp.real(reference(base, source, coefficient, beta)).sum()
        np.testing.assert_allclose(jax.jacfwd(jax.grad(loss))(alpha),
                                   jax.jacfwd(jax.grad(expected_loss))(alpha), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("static_factor", [None, .5])
@pytest.mark.parametrize("coefficient", [0., 1., 2.])
def test_coefficient_branches_keep_algebraic_first_and_second_derivatives(static_factor, coefficient):
    with jax.enable_x64():
        source, base = jnp.asarray([1.25, 3.5, -2.75], dtype=jnp.float32), jnp.arange(7, dtype=jnp.float32)
        arguments = (base, source, jnp.float64(coefficient), jnp.float64(1))
        directions = tuple(jnp.ones_like(value) for value in arguments)

        def operation(old, new, alpha, beta):
            effective = alpha if static_factor is None else alpha * jnp.float64(static_factor)
            return update_p.bind(new, old, effective, beta, records=PARTIAL)

        def reference(old, new, alpha, beta):
            effective = alpha if static_factor is None else alpha * jnp.float64(static_factor)
            return old.at[1::2].set((effective * new + beta * old[1::2]).astype(old.dtype))

        def derivatives(function, *values):
            first = lambda *items: jax.jvp(function, items, directions)[1]
            return (first(*values), jax.grad(lambda *items: jnp.sum(function(*items)), argnums=(0, 1, 2, 3))(*values),
                    jax.jvp(first, values, directions)[1])

        actual = jax.jit(lambda *values: derivatives(operation, *values))(*arguments)
        expected = derivatives(reference, *arguments)
        for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(result, wanted, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("factor", [np.float64(.125), np.complex128(1 + 2j)])
def test_mapping_final_cast_preserves_derivatives(factor):
    with jax.enable_x64():
        source = jnp.asarray([1.25, 2.5, -3.75], dtype=jnp.float32)
        native = lambda value: accumulation_p.bind(value, jnp.asarray(factor), records=PARTIAL,
                                                   coefficient_records=(0,), output_size=7, dtype=value.dtype)
        reference = lambda value: jnp.zeros(7, dtype=value.dtype).at[1::2].set(jnp.real(factor * value).astype(value.dtype))
        for result, wanted in zip(jax.jvp(native, (source,), (jnp.ones_like(source),)),
                                   jax.jvp(reference, (source,), (jnp.ones_like(source),)), strict=True):
            assert_components(result, wanted)
        cotangent = jnp.ones(7, dtype=source.dtype)
        assert_components(jax.vjp(native, source)[1](cotangent)[0], jax.vjp(reference, source)[1](cotangent)[0])
