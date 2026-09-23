"""JAX batching and nested mapped-axis contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import accumulation_p, copy_p, reduction_p, update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.availability import native_available
from tests.stride.support.layouts import PARTIAL
from tests.stride.support.oracles.affine_ad import (
    PARTITIONS as AFFINE_AD_PARTITIONS,
    assert_close,
    assert_native_calls,
    reference_transform,
)
from tests.stride.support.oracles.copy import (
    PARTITIONS as COPY_PARTITIONS,
    assert_single_native_call,
)
from tests.stride.support.oracles.raw_update import COMPLETE, reference_update
from tests.stride.support.oracles.reduction import execute, reference


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_vmap_batches_affine_derivatives_without_unrolling_records(mode):
    factors = (jnp.float32(1.25), jnp.float32(-.5))
    native = lambda value: accumulation_p.bind(
        value, *factors, records=AFFINE_AD_PARTITIONS, coefficient_records=(0, 1), output_size=16, dtype=np.dtype(jnp.float32),
    )
    reference = lambda value: reference_transform(value, AFFINE_AD_PARTITIONS, factors, 16, jnp.float32)
    source = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    direction = jnp.linspace(-2, 3, 48).reshape(3, 16)
    if mode == "jvp":
        derivative = lambda operation, value, tangent: jax.jvp(operation, (value,), (tangent,))
    else:
        derivative = lambda operation, value, tangent: jax.vjp(operation, value)[1](tangent)[0]
    compiled = jax.jit(jax.vmap(lambda value, tangent: derivative(native, value, tangent)))
    actual = compiled(source, direction)
    expected = jax.vmap(lambda value, tangent: derivative(reference, value, tangent))(source, direction)
    for result, wanted in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_close(result, wanted)
    assert_native_calls(compiled.lower(source, direction), 2 if mode == "jvp" else 1)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_multiple_records_and_outer_vmap():
    records = (AffineRecord((2,), (1,), 0, (1,), 0), AffineRecord((2,), (-1,), 1, (1,), 2))
    run = lambda values: copy_p.bind(values, records=records, output_size=5, dtype=jnp.dtype("float16"))
    reference = lambda values: jnp.concatenate((values[..., [0, 1, 1, 0]].astype(jnp.float16),
                                               jnp.zeros((*values.shape[:-1], 1), dtype=jnp.float16)), axis=-1)
    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    cotangent = jnp.ones((2, 5), dtype=jnp.float16)
    np.testing.assert_array_equal(jax.jit(jax.vmap(run))(source), reference(source))
    actual = jax.jit(jax.vjp(jax.vmap(run), source)[1])(cotangent)[0]
    np.testing.assert_array_equal(actual, jax.vjp(reference, source)[1](cotangent)[0])


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("mode", ["leading", "trailing", "nested"])
def test_vmap_axes_use_one_batched_native_call(mode):
    apply = lambda value: copy_p.bind(value, records=COPY_PARTITIONS, output_size=16, dtype=value.dtype)
    indices = np.arange(16).reshape(4, 4)[:, [2, 3, 0, 1]].ravel()
    if mode == "leading":
        source = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
        operation = jax.jit(jax.vmap(apply))
        expected = np.asarray(source)[:, indices]
    elif mode == "trailing":
        source = jnp.arange(48, dtype=jnp.float32).reshape(16, 3)
        operation = jax.jit(jax.vmap(apply, in_axes=1, out_axes=1))
        expected = np.asarray(source)[indices, :]
    else:
        source = jnp.arange(96, dtype=jnp.float32).reshape(16, 2, 3)
        operation = jax.jit(jax.vmap(jax.vmap(apply, in_axes=1, out_axes=1), in_axes=2, out_axes=2))
        expected = np.asarray(source)[indices, :, :]
    np.testing.assert_array_equal(operation(source), expected)
    assert_single_native_call(operation.lower(source))


def test_multirecord_mapping_and_vmap_share_one_update_primitive():
    records = (AffineRecord((4, 2), (4, 1), 0, (4, 1), 2),
               AffineRecord((4, 2), (4, 1), 2, (4, 1), 0))
    base = jnp.arange(48, dtype=jnp.float32).reshape(3, 16)
    source = base + 2
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    function = lambda old, new, factor: update_p.bind(
        new, old, factor, jnp.int32(1), records=records,
    )
    batched = jax.jit(function)(base, source, factors)
    mapped = jax.jit(jax.vmap(function))(base, source, factors)
    np.testing.assert_array_equal(batched, mapped)
    primitives = jax.make_jaxpr(function)(base, source, factors).jaxpr.eqns
    assert [equation.primitive.name for equation in primitives].count(
        "tensor0_stride_update") == 1
    selected = source.reshape(3, 4, 4)[..., [2, 3, 0, 1]].reshape(3, 16)
    np.testing.assert_array_equal(batched, factors[:, None] * selected + base)


@pytest.mark.parametrize("source_dtype,result_dtype", [(jnp.float32, jnp.float32),
                                                      (jnp.float16, jnp.float32),
                                                      (jnp.complex64, jnp.complex64)])
def test_complete_contiguous_updates_use_each_batch_coefficient(source_dtype, result_dtype):
    records = (AffineRecord((4,), (1,), 0, (1,), 0),)
    base = jnp.asarray([[jnp.nan] * 4, [10] * 4, [7] * 4], dtype=result_dtype)
    source = jnp.asarray([[1] * 4, [jnp.nan] * 4, [2] * 4], dtype=source_dtype)
    source_factor = jnp.asarray([1, 0, 2], dtype=result_dtype)
    base_factor = jnp.asarray([0, 1, 3], dtype=result_dtype)
    function = lambda old, new, lhs, rhs: update_p.bind(
        new, old, lhs, rhs, records=records,
    )
    expected = jnp.asarray([[1] * 4, [10] * 4, [25] * 4], dtype=result_dtype)
    for operation in (jax.jit(function), jax.jit(jax.vmap(function))):
        np.testing.assert_array_equal(operation(base, source, source_factor, base_factor),
                                      expected)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("outer_size", [0, 3])
def test_reduction_batches_outer_vmap_with_shared_storage_batch_coefficients_and_ad(outer_size):
    factors = jnp.asarray([0, 1, 2], dtype=jnp.float32)
    source = jnp.ones((outer_size, 3, 3), dtype=jnp.float32)

    def execute(data):
        return reduction_p.bind(
            data, factors, coefficient_records=(0,),
            records=(AffineRecord((3,), (1,), 0, (1,), 0),),
            output_shapes=((1,),), reduction_axes=((True,),), output_size=1, dtype=data.dtype)

    mapped = jax.vmap(execute)
    result, tangent = jax.jit(lambda data: jax.jvp(mapped, (data,), (data,)))(source)
    expected = np.broadcast_to(np.asarray([[0], [3], [6]]), (outer_size, 3, 1))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(tangent, expected)
    gradient = jax.jit(jax.grad(lambda data: mapped(data).sum()))(source)
    np.testing.assert_array_equal(gradient, np.broadcast_to(np.asarray(factors)[:, None], source.shape))


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.skipif(not native_available(), reason='native CPU stride unavailable')
@pytest.mark.parametrize("batch_size", [0, 3])
@pytest.mark.parametrize("source_axis,coefficient_axis", [(1, None), (None, 0), (1, 0)])
def test_vmap_shared_source_and_coefficients(batch_size, source_axis, coefficient_axis):
    source = jnp.arange(8, dtype=jnp.float32) if source_axis is None else jnp.arange(
        8 * batch_size, dtype=jnp.float32).reshape((8, batch_size))
    first = jnp.float32(2) if coefficient_axis is None else jnp.arange(batch_size, dtype=jnp.float32)
    second = jnp.float32(3)
    arguments = source, first, second
    run, oracle = (jax.vmap(lambda data, alpha, beta: execute(data, jnp.float32(1), alpha, beta),
                           in_axes=(source_axis, coefficient_axis, None)),
                   jax.vmap(lambda data, alpha, beta: reference(data, jnp.float32(1), alpha, beta),
                            in_axes=(source_axis, coefficient_axis, None)))
    cotangent = jnp.ones((batch_size, 7), dtype=jnp.float32)
    for actual, expected in zip(jax.jit(jax.vjp(run, *arguments)[1])(cotangent),
                                jax.vjp(oracle, *arguments)[1](cotangent), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("operation", ["reduction", "accumulation"])
@pytest.mark.parametrize("batch_count", [0, 1, 3])
@pytest.mark.parametrize("row_batch", [(), (0,), (2,)])
def test_empty_and_nested_batches(operation, batch_count, row_batch):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tests.stride.support.oracles.reduction import functions

    execute, reference = functions(operation, "float32")
    host = (jnp.ones((batch_count, *row_batch, 5)),
            jnp.arange(batch_count, dtype=jnp.float32).reshape(batch_count, 1), jnp.float32(.5))
    run, oracle = (jax.vmap(function, in_axes=(0, 0, None)) for function in (execute, reference))
    for actual, expected in zip(jax.jit(lambda *inputs: jax.jvp(run, inputs, inputs))(*host),
                                jax.jvp(oracle, host, host), strict=True):
        np.testing.assert_allclose(actual, expected)
    cotangent = jnp.ones((batch_count, *row_batch, 6))
    for actual, expected in zip(jax.jit(lambda *inputs: jax.vjp(run, *inputs)[1](cotangent))(*host),
                                jax.vjp(oracle, *host)[1](cotangent), strict=True):
        np.testing.assert_allclose(actual, expected)
    outer_host = tuple(jnp.stack((value, value)) for value in host)
    np.testing.assert_allclose(jax.jit(jax.vmap(run))(*outer_host), jax.vmap(oracle)(*outer_host))


@pytest.mark.parametrize("operation", ["reduction", "accumulation"])
@pytest.mark.parametrize("invalid", ["source", "coefficient"])
def test_invalid_slice_shapes(operation, invalid):
    import jax
    import jax.numpy as jnp
    from tests.stride.support.oracles.reduction import functions

    run, _ = functions(operation, "float32")
    source = jnp.ones((2,)) if invalid == "source" else jnp.ones((2, 5))
    factor = jnp.ones((2, 2)) if invalid == "coefficient" else jnp.ones((2,))
    with pytest.raises(ValueError, match="storage.*at least one dimension|coefficients must"):
        jax.jit(jax.vmap(run, in_axes=(0, 0, None)))(source, factor, jnp.float32(1))


@pytest.mark.parametrize("operation", ["reduction", "accumulation"])
def test_zero_mapped_coefficients_skip_nonfinite_rows(operation):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from tests.stride.support.oracles.reduction import functions

    execute, reference = functions(operation, "float32")
    finite = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)
    source = finite.at[0].set(jnp.nan).at[3].set(jnp.inf)
    first = jnp.asarray([[0], [1], [2], [0]], jnp.float32)
    second = jnp.broadcast_to(first, (4, 3))
    run, oracle = jax.vmap(execute), jax.vmap(reference)
    np.testing.assert_allclose(jax.jit(run)(source, first, second), oracle(finite, first, second))


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("records,source_size,output_size", [(PARTIAL, 32, 50), (COMPLETE, 16, 16)])
@pytest.mark.parametrize("in_axes", [(0, 0), (0, None), (None, 0)])
def test_update_vmap_operand_combinations(beta, records, source_size, output_size, in_axes):
    base, source = jnp.arange(output_size, dtype=jnp.float32), jnp.arange(source_size, dtype=jnp.float32)
    operands = (jnp.stack((base, 2 * base, -base)) if in_axes[0] == 0 else base,
                jnp.stack((source, 3 * source, -source)) if in_axes[1] == 0 else source)
    operation = lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=records)
    reference = lambda old, new: reference_update(new, old, records, .5, beta)
    np.testing.assert_allclose(jax.jit(jax.vmap(operation, in_axes=in_axes))(*operands),
                               jax.vmap(reference, in_axes=in_axes)(*operands), rtol=2e-6, atol=1e-6)
