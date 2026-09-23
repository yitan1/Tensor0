"""JAX AD reduction contracts."""

from concurrent.futures import ThreadPoolExecutor
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import reduction_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.oracles.reduction import (
    CROSS,
    CROSS_SHAPE_TYPES,
    assert_close,
    compare_derivatives,
    execute,
    functions,
    operands,
    reference,
    reference_sum,
)
from tests.stride.support.samples import values


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float16, jnp.float32), (jnp.bfloat16, jnp.float32),
    (jnp.float16, jnp.float16), (jnp.bfloat16, jnp.bfloat16),
    (jnp.float32, jnp.float32), (jnp.float64, jnp.float64), (jnp.complex64, jnp.complex64),
    (jnp.complex128, jnp.complex128), (jnp.float32, jnp.complex64), (jnp.complex64, jnp.float32)])
def test_mixed_reduction_forward_coefficient_ad_batch_and_nested_transpose(source_dtype, result_dtype):
    with jax.enable_x64():
        parameters = dict(records=(AffineRecord((4, 3), (3, 1), 0, (3, 1), 0),),
                          output_shapes=((1, 3),), reduction_axes=((True, False),), output_size=3, dtype=np.dtype(result_dtype))
        source = (jnp.arange(12, dtype=jnp.float32) / 8 - .5).astype(source_dtype)
        if jnp.issubdtype(source_dtype, jnp.complexfloating):
            source = source * (1 + .5j)
        factor = jnp.asarray(.75 - .5j if jnp.issubdtype(result_dtype, jnp.complexfloating) else -.75,
                             dtype=result_dtype)
        native = lambda value, coefficient: reduction_p.bind(value, coefficient, coefficient_records=(0,), **parameters)
        reference = lambda value, coefficient: reference_sum(value, (coefficient,), **parameters)
        cotangent = jnp.ones(3, dtype=result_dtype) * .5
        directions = (jnp.ones_like(source) * .25, jnp.ones_like(factor) * .5)
        for function in (lambda operation: jax.jvp(operation, (source, factor), directions),
                         lambda operation: jax.vjp(operation, source, factor)[1](cotangent)):
            actual, expected = function(native), function(reference)
            for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                assert_close(result, wanted)
        sources = jnp.stack((source, source * 2))
        assert_close(jax.jit(jax.vmap(native, in_axes=(0, None)))(sources, factor),
                     jax.vmap(reference, in_axes=(0, None))(sources, factor))
        pullback = lambda cot: jax.vjp(lambda value: native(value, factor), source)[1](cot)[0]
        expected_pullback = lambda cot: jax.vjp(lambda value: reference(value, factor), source)[1](cot)[0]
        assert_close(jax.jit(lambda cot: jax.linear_transpose(pullback, cotangent)(cot)[0])(jnp.ones_like(source)),
                     jax.linear_transpose(expected_pullback, cotangent)(jnp.ones_like(source))[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("axes", [(False, False), (False, True)])
def test_overlapping_records_different_fibers_and_output_selections(axes):
    reduced = axes[1]
    records = (AffineRecord((4, 2 if reduced else 1), (2, 1), 0, (-2, 1), 7),
               AffineRecord((4, 3 if reduced else 1), (3, 1), 8, (-2, 1), 7),
               AffineRecord((2, 1), (1, 1), 20, (2, 1), 2))
    parameters = dict(records=records, output_shapes=((4, 1), (4, 1), (2, 1)),
                      reduction_axes=(axes, axes, axes), output_size=10, dtype=np.dtype(jnp.float32))
    factors = (jnp.float32(1.25), jnp.float32(-.5), jnp.float32(2))
    native = lambda value: reduction_p.bind(value, *factors, coefficient_records=(0, 1, 2), **parameters)
    reference = lambda value: reference_sum(value, factors, **parameters)
    source = jnp.arange(22, dtype=jnp.float32) / 4
    compiled = jax.jit(native).lower(source).compile()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda shift: compiled(source + shift), range(4)))
    for shift, actual in enumerate(outputs):
        assert_close(actual, reference(source + shift))
    cotangent = jnp.linspace(-1, 1, 10)
    assert_close(jax.jit(lambda cot: jax.vjp(native, source)[1](cot)[0])(cotangent),
                 jax.vjp(reference, source)[1](cotangent)[0])
    assert_close(jax.jit(jax.vmap(native))(jnp.stack((source, source * 2))),
                 jax.vmap(reference)(jnp.stack((source, source * 2))))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mixed_reduction_reverse_product_range_and_coefficient_ad_boundary():
    with jax.enable_x64():
        source = jnp.asarray([.001], dtype=jnp.float64)
        record = AffineRecord((1,), (1,), 0, (1,), 0)
        def execute(values, factor):
            return reduction_p.bind(values, factor, records=(record,), output_shapes=((1,),),
                reduction_axes=((True,),), coefficient_records=(0,), output_size=1, dtype=jnp.dtype("float16"))
        run = lambda values: execute(values, jnp.float16(1000))
        actual = jax.jit(jax.vjp(run, source)[1])(jnp.asarray([1000], dtype=jnp.float16))[0]
        assert actual.dtype == source.dtype
        np.testing.assert_allclose(actual, [1_000_000], rtol=1e-14, atol=0)
        _, tangent = jax.jvp(lambda factor: execute(source, factor), (jnp.float16(2),), (jnp.float16(1),))
        np.testing.assert_array_equal(tangent, source.astype(jnp.float16))


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_mixed_reduction_batch_coefficients_and_zero_one_branches():
    source = jnp.ones((3, 4), dtype=jnp.float16)
    record = AffineRecord((2, 2), (1, 1), 0, (1, 1), 1)
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    def execute(values, coefficients):
        return reduction_p.bind(values, coefficients, records=(record,), output_shapes=((2, 1),),
            reduction_axes=((False, True),), coefficient_records=(0,), output_size=4, dtype=jnp.dtype("float32"))
    run = lambda values: execute(values, factors)
    cotangent = jnp.asarray([[jnp.nan, 1, 2, jnp.nan]] * 3, dtype=jnp.float32)
    actual = jax.jit(jax.vjp(run, source)[1])(cotangent)[0]
    assert actual.dtype == source.dtype
    np.testing.assert_allclose(actual, [[0, 0, 0, 0], [1, 3, 2, 0], [2, 6, 4, 0]], rtol=1e-3, atol=1e-3)
    source_rows = jnp.stack((source, source * 2))
    factor_rows = jnp.stack((factors, factors))
    mapped = jax.vmap(execute)
    expected = jnp.stack((run(source), run(source * 2)))
    np.testing.assert_allclose(jax.jit(mapped)(source_rows, factor_rows), expected, rtol=1e-3, atol=1e-3)
    _, tangent = jax.jvp(run, (source,), (jnp.full_like(source, jnp.inf),))
    np.testing.assert_array_equal(tangent[0], [0, 0, 0, 0])
    gradient = jax.linear_transpose(run, source)(jnp.full((3, 4), jnp.inf, dtype=jnp.float32))[0]
    np.testing.assert_array_equal(gradient[0], [0, 0, 0, 0])
    assert jnp.all(jnp.isposinf(gradient[1, :3]))
    assert gradient[1, 3] == 0


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
# Retain both narrow kinds at representative shapes and wide nonempty AD.
@pytest.mark.parametrize("dtype,batch_shape,shape_kind",
                         [(dtype, batch, form) for dtype in ("float32", "complex64")
                          for batch, form in (((), "scalar"), ((), "one"), ((2,), "batch"),
                                              ((2, 3), "scalar"), ((2, 3), "one"),
                                              ((2, 3), "batch"), ((0,), "scalar"),
                                              ((2, 0), "batch"))]
                         + [(dtype, (2, 3), "batch") for dtype in ("float64", "complex128")])
def test_joint_jvp_and_vjp(dtype, batch_shape, shape_kind):
    shape = {"scalar": (), "one": (1,), "batch": batch_shape}[shape_kind]
    arguments = operands(dtype, batch_shape, shape)
    directions = tuple(jnp.full_like(value, 1 + 2j if dtype in ("complex64", "complex128") else 2) for value in arguments)
    for actual, expected in zip(jax.jit(lambda *values: jax.jvp(execute, values, directions))(*arguments),
                                jax.jvp(reference, arguments, directions), strict=True):
        np.testing.assert_array_equal(actual, expected)
    cotangent = (jnp.arange(prod(batch_shape) * 7) % 7).astype(dtype).reshape((*batch_shape, 7))
    if dtype in ("complex64", "complex128"):
        cotangent = cotangent * (1 - 1j)
    for actual, expected in zip(jax.jit(jax.vjp(execute, *arguments)[1])(cotangent),
                                jax.vjp(reference, *arguments)[1](cotangent), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("active", [(0,), (1,), (2,), (3,), (1, 2, 3), (1, 3)])
def test_selected_tangents_and_direct_transpose(dtype, active):
    arguments = operands(dtype, (2, 3), ())
    def bind(function, *values):
        operands = list(arguments)
        for index, value in zip(active, values, strict=True):
            operands[index] = value
        return function(*operands)
    run, oracle = lambda *values: bind(execute, *values), lambda *values: bind(reference, *values)
    primals = tuple(arguments[index] for index in active)
    tangents = tuple(jnp.ones_like(value) for value in primals)
    np.testing.assert_array_equal(jax.jvp(run, primals, tangents)[1], jax.jvp(oracle, primals, tangents)[1])
    cotangent = jnp.broadcast_to(jnp.arange(7).astype(dtype), (2, 3, 7))
    expected = jax.vjp(oracle, *primals)[1](cotangent)
    for gradients in (jax.jit(jax.vjp(run, *primals)[1])(cotangent),
                      jax.jit(jax.linear_transpose(run, *primals))(cotangent)):
        for actual, value in zip(gradients, expected, strict=True):
            np.testing.assert_array_equal(actual, value)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_coefficient_only_lowering_and_unselected_nonfinite_values():
    record = AffineRecord((2, 2), (2, 1), 1, (2, 100), 1)
    def run(data, factor):
        return reduction_p.bind(data, factor, records=(record,), output_shapes=((2, 1),),
            reduction_axes=((False, True),), coefficient_records=(0,), output_size=5, dtype=data.dtype)
    source = jnp.asarray([np.nan, 2, 3, 4, 5, np.inf], jnp.float32)
    cotangent = jnp.asarray([np.nan, 2, np.inf, 3, np.nan], jnp.float32)
    reverse = jax.jit(lambda data, value: jax.vjp(lambda factor: run(data, factor), jnp.float32(0))[1](value)[0])
    np.testing.assert_array_equal(reverse(source, cotangent), 37)
    text = reverse.lower(source, cotangent).as_text()
    assert text.count("stablehlo.custom_call @tensor0_stride_dot_f32_cpu_v1") == 1
    for operation in ("tensor0_stride_reduction", "tensor0_stride_accumulation", "stablehlo.gather", "stablehlo.scatter"):
        assert operation not in text


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,), (2, 0)])
def test_empty_input_and_symbolic_zero(batch_shape):
    source = jnp.zeros((*batch_shape, 0), jnp.float32)
    def run(data, factor):
        return reduction_p.bind(data, factor, records=(AffineRecord((0,), (1,), 0, (1,), 1),),
            output_shapes=((1,),), reduction_axes=((True,),), coefficient_records=(0,), output_size=3, dtype=data.dtype)
    arguments = source, jnp.float32(2)
    output, pullback = jax.vjp(run, *arguments)
    np.testing.assert_array_equal(output, jnp.zeros((*batch_shape, 3)))
    source_gradient, factor_gradient = jax.jit(pullback)(jnp.ones_like(output))
    np.testing.assert_array_equal(source_gradient, source)
    np.testing.assert_array_equal(factor_gradient, 0)
    frozen = lambda *values: run(*(jax.lax.stop_gradient(value) for value in values))
    np.testing.assert_array_equal(jax.jvp(frozen, arguments, tuple(jnp.ones_like(value) for value in arguments))[1], output)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_joint_bilinear_direct_transpose_remains_rejected():
    source, empty, first, second = operands("float32", (), ())
    with pytest.raises(NotImplementedError, match="known source or known coefficients"):
        jax.linear_transpose(lambda data, value: execute(data, empty, value, second), source, first)(jnp.ones(7, dtype=source.dtype))


# Six mixed-kind AD patterns plus low-width forward representatives; empty
# shapes retain two explicit patterns rather than the full precision cross.
CROSS_AD_CASES = [(*types, (2, 3), "batch") for types in CROSS
                  if types in CROSS_SHAPE_TYPES or all(t in ("float32", "complex64") for t in types)] + [
    (*types, batch, form) for types in CROSS_SHAPE_TYPES
    for batch, form in (((), "scalar"), ((2, 3), "one"),
                        ((0,), "scalar"), ((2, 0), "batch"))
    if batch in ((), (2, 3))
    or (types, batch) in ((("complex128", "float32", "float64"), (0,)),
                         (("float64", "float32", "complex128"), (2, 0)))
]


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype,batch_shape,shape_kind", CROSS_AD_CASES)
def test_cross_kind_contract(source_dtype, coefficient_dtype, dtype, batch_shape, shape_kind):
    source = values((*batch_shape, 5), source_dtype)
    shape = {"scalar": (), "one": (1,), "batch": batch_shape}[shape_kind]
    first = jnp.full(shape, 2 + 1j if coefficient_dtype.startswith("complex") else 2, dtype=coefficient_dtype)
    second = jnp.full(batch_shape, .5 + .5j if dtype.startswith("complex") else .5, dtype=dtype)
    run, reference = functions("reduction", dtype)
    arguments = (source, first, second)
    if (source_dtype, coefficient_dtype, dtype) in CROSS_SHAPE_TYPES:
        compare_derivatives(run, reference, arguments)
    else:
        actual, expected = jax.jit(run)(*arguments), reference(*arguments)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("source_dtype,coefficient_dtype,dtype", [
    ("float32", "complex128", "float64"), ("complex64", "float64", "complex128"),
    ("complex128", "complex64", "float32"), ("float64", "float32", "complex64")])
@pytest.mark.parametrize("factor", [0, 1, 2])
def test_higher_derivatives_and_vmap(source_dtype, coefficient_dtype, dtype, factor):
    run, reference = functions("reduction", dtype)
    source, coefficient = values((5,), source_dtype), jnp.asarray(factor, dtype=coefficient_dtype)
    fixed = jnp.asarray(.5, dtype=dtype)
    def objective(function, data, alpha):
        output = function(data, alpha, fixed)
        return jnp.real(jnp.sum(output * jnp.conj(output)))
    gradient = jax.grad(lambda *inputs: objective(run, *inputs), argnums=(0, 1))
    expected_gradient = jax.grad(lambda *inputs: objective(reference, *inputs), argnums=(0, 1))
    arguments = source, coefficient
    directions = jnp.ones_like(source), jnp.ones_like(coefficient)
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(gradient, inputs, directions)[1])(*arguments),
                                jax.jvp(expected_gradient, arguments, directions)[1], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    for actual, expected in zip(jax.jit(jax.vjp(gradient, *arguments)[1])(directions),
                                jax.vjp(expected_gradient, *arguments)[1](directions), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    compare_derivatives(jax.vmap(run, in_axes=(0, 0, None)), jax.vmap(reference, in_axes=(0, 0, None)),
                        (jnp.stack((source, source * 2)), jnp.stack((coefficient, coefficient)), fixed))


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
def test_projection_after_multiplication():
    record = AffineRecord((2,), (0,), 0, (1,), 1)
    def run(source, factor):
        return reduction_p.bind(source, factor, records=(record,), output_shapes=((1,),),
            reduction_axes=((True,),), coefficient_records=(0,), output_size=3, dtype=jnp.dtype("complex128"))
    source, factor = jnp.asarray([2], jnp.float32), jnp.complex128(3 + 4j)
    cotangent = jnp.asarray([0, 1 + 2j, 0], jnp.complex128)
    gradients = jax.jit(jax.vjp(run, source, factor)[1])(cotangent)
    np.testing.assert_array_equal(gradients[0], [-10])
    np.testing.assert_array_equal(gradients[1], 4 + 8j)
    text = jax.jit(lambda data, alpha, value: jax.vjp(run, data, alpha)[1](value)).lower(source, factor, cotangent).as_text()
    assert "tensor0_stride_accumulation_" in text and "tensor0_stride_dot_" in text
    for name in ("stablehlo.real", "stablehlo.gather", "stablehlo.scatter", "tensor0_stride_update"):
        assert name not in text
