from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import SU2Irrep, hom, space
from tensor0._stride import StridedView, materialize
from tensor0._stride._ffi._calls import execute_copy
from tensor0._stride._ffi._descriptor import encode_layout
from tensor0._stride._layout import AffineRecord
from tensor0.structure import get_degeneracystructure

from ._support import native_available


def reference_materialize(storage, sizes, strides, offset):
    indices = np.full(sizes, offset, dtype=np.int32)
    for axis, (size, stride) in enumerate(zip(sizes, strides, strict=True)):
        shape = (1,) * axis + (size,) + (1,) * (len(sizes) - axis - 1)
        indices += np.arange(size, dtype=np.int32).reshape(shape) * stride
    return storage[..., indices]


def _view(
    data,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> StridedView:
    return StridedView(data, sizes, strides, offset)


def test_static_contiguous_materialization_uses_native_on_cpu() -> None:
    source = jnp.arange(20, dtype=jnp.float32)
    function = jax.jit(
        lambda value: materialize(
            _view(value, (2, 3), (3, 1), 4),
        )
    )

    actual = function(source)
    hlo = str(function.lower(source).compiler_ir("stablehlo")).lower()

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(source[4:10].reshape(2, 3)),
    )
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_static_sliced_permuted_materialization_and_cast_use_native() -> None:
    source = jnp.arange(20, dtype=jnp.float32)
    expected = reference_materialize(source, (2, 3), (1, 4), 2).astype(
        jnp.complex64
    )
    function = jax.jit(
        lambda value: materialize(
            _view(value, (2, 3), (1, 4), 2),
            dtype=jnp.complex64,
        )
    )

    actual = function(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.dtype == jnp.complex64
    hlo = str(function.lower(source).compiler_ir("stablehlo")).lower()
    assert "tensor0_stride_copy_c64_cpu_v1" in hlo
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo


def test_static_empty_materialization_preserves_shape_and_source_dtype() -> None:
    source = jnp.arange(20, dtype=jnp.float32)

    actual = materialize(
        _view(source, (2, 0, 3), (3, 3, 1), 5),
    )

    assert actual.shape == (2, 0, 3)
    assert actual.dtype == source.dtype


def test_static_empty_materialization_preserves_explicit_result_dtype() -> None:
    source = jnp.arange(20, dtype=jnp.float32)

    actual = materialize(
        _view(source, (2, 0, 3), (3, 3, 1), 5),
        dtype=jnp.complex64,
    )

    assert actual.shape == (2, 0, 3)
    assert actual.dtype == jnp.complex64


def test_materialize_requires_a_bound_view() -> None:
    source = np.arange(20, dtype=np.float32)

    with pytest.raises(TypeError, match="materialize requires a StridedView"):
        materialize(source)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="data must be a JAX Array"):
        StridedView(source, (2, 3), (1, 4), 2)


def test_small_concrete_materialization_matches_compiled_copy() -> None:
    source = jnp.arange(20, dtype=jnp.float32)
    actual = materialize(
        _view(source, (2, 3), (1, 4), 2),
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(reference_materialize(source, (2, 3), (1, 4), 2)),
    )
    compiled = jax.jit(lambda value: materialize(_view(value, (2, 3), (1, 4), 2)))
    np.testing.assert_array_equal(actual, compiled(source))
    assert "tensor0_stride_copy_f32_cpu_v1" in str(compiled.lower(source).compiler_ir("stablehlo"))


def test_materialize_jit_jvp_vjp_linear_transpose_and_vmap_match_reference() -> None:
    source = jnp.linspace(-1, 1, 20, dtype=jnp.float32)
    tangent = jnp.linspace(1, 2, 20, dtype=jnp.float32)
    cotangent = jnp.linspace(-2, 1, 6, dtype=jnp.float32).reshape(2, 3)

    def migrated(value):
        return materialize(
            _view(value, (2, 3), (1, 4), 2),
        )

    def reference(value):
        return reference_materialize(value, (2, 3), (1, 4), 2)

    migrated_primal, migrated_tangent = jax.jvp(
        migrated,
        (source,),
        (tangent,),
    )
    reference_primal, reference_tangent = jax.jvp(
        reference,
        (source,),
        (tangent,),
    )
    migrated_vjp = jax.vjp(migrated, source)[1](cotangent)[0]
    reference_vjp = jax.vjp(reference, source)[1](cotangent)[0]
    migrated_transpose = jax.linear_transpose(
        migrated,
        jnp.zeros_like(source),
    )(cotangent)[0]
    reference_transpose = jax.linear_transpose(
        reference,
        jnp.zeros_like(source),
    )(cotangent)[0]
    batch = jnp.stack((source, 2 * source, -source))

    np.testing.assert_array_equal(migrated_primal, reference_primal)
    np.testing.assert_array_equal(migrated_tangent, reference_tangent)
    np.testing.assert_array_equal(migrated_vjp, reference_vjp)
    np.testing.assert_array_equal(migrated_transpose, reference_transpose)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(batch),
        jax.jit(jax.vmap(reference))(batch),
    )


def test_mixed_materialize_jit_jvp_vjp_and_vmap_match_explicit_cast() -> None:
    source = jnp.linspace(-1, 1, 6, dtype=jnp.float32)
    tangent = jnp.linspace(1, 2, 6, dtype=jnp.float32)
    cotangent = (
        jnp.linspace(-2, 1, 6, dtype=jnp.float32)
        + 1j * jnp.linspace(3, -1, 6, dtype=jnp.float32)
    ).reshape(2, 3)

    def migrated(value):
        return materialize(
            _view(value, (2, 3), (3, 1), 0),
            dtype=jnp.complex64,
        )

    def explicit(value):
        return reference_materialize(value, (2, 3), (3, 1), 0).astype(
            jnp.complex64
        )

    actual_primal, actual_tangent = jax.jvp(migrated, (source,), (tangent,))
    expected_primal, expected_tangent = jax.jvp(explicit, (source,), (tangent,))
    actual_vjp = jax.vjp(migrated, source)[1](cotangent)[0]
    expected_vjp = jax.vjp(explicit, source)[1](cotangent)[0]
    batch = jnp.stack((source, 2 * source, -source))

    np.testing.assert_array_equal(actual_primal, expected_primal)
    np.testing.assert_array_equal(actual_tangent, expected_tangent)
    np.testing.assert_array_equal(actual_vjp, expected_vjp)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(batch),
        jax.jit(jax.vmap(explicit))(batch),
    )


def test_subblock_metadata_can_be_materialized_directly() -> None:
    factor = space(SU2Irrep, {0: 2, 1: 3})
    target = hom((factor, factor), (factor, factor))
    layout = get_degeneracystructure(target)
    subblock = layout.subblockstructure[0]
    source = jnp.arange(layout.total_dim, dtype=jnp.float32)
    expected = reference_materialize(
        source,
        tuple(subblock.sizes),
        tuple(subblock.strides),
        subblock.offset,
    )
    actual = materialize(
        _view(
            source,
            tuple(subblock.sizes),
            tuple(subblock.strides),
            subblock.offset,
        ),
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
@pytest.mark.parametrize("side", [64, 512])
def test_partial_materialization_uses_native_copy(side) -> None:
    sizes = (side, side)
    strides = (1, side)
    copied = sizes[0] * sizes[1]
    source = jnp.arange(copied + 1, dtype=jnp.float32)
    function = jax.jit(
        lambda value: materialize(
            _view(value, sizes, strides, 0),
        )
    )

    hlo = str(function.lower(source).compiler_ir("stablehlo")).lower()
    actual = function(source)
    actual.block_until_ready()

    assert "tensor0_stride_copy_f32_cpu_v1" in hlo
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(source[:-1].reshape(sizes).T),
    )

    eager = materialize(
        _view(source, sizes, strides, 0),
    )
    eager.block_until_ready()

    np.testing.assert_array_equal(eager, actual)


@pytest.mark.parametrize("sizes,strides,offset", [
    ((2, 3), (-1, 4), 3),
    ((2, 3), (0, 1), 2),
    ((2, 2), (1, 1), 1),
    ((), (), 2),
    ((2, 0), (3, 1), 0),
])
@pytest.mark.parametrize("batch_count", [0, 1, 3])
def test_materialize_address_layouts_and_batches(sizes, strides, offset, batch_count) -> None:
    source = jnp.arange(batch_count * 16, dtype=jnp.float32).reshape(batch_count, 16)
    view = StridedView(source, sizes, strides, offset)
    expected = reference_materialize(source, sizes, strides, offset)
    actual = jax.jit(materialize)(view)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (batch_count, *sizes)


@pytest.mark.parametrize("strides", [(0, 1), (1, 1)])
def test_materialize_repeated_reads_accumulate_source_gradient(strides) -> None:
    source = jnp.arange(8, dtype=jnp.float32)
    cotangent = jnp.array([[1., 2.], [3., 4.]])

    def operation(data):
        return materialize(StridedView(data, (2, 2), strides, 1))

    def reference(data):
        return reference_materialize(data, (2, 2), strides, 1)

    actual = jax.jit(lambda data: jax.vjp(operation, data)[1](cotangent)[0])(source)
    expected = jax.vjp(reference, source)[1](cotangent)[0]
    np.testing.assert_array_equal(actual, expected)


CONVERSION_DTYPES = (
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "float16", "bfloat16", "float32", "float64", "complex64", "complex128",
)



@pytest.mark.parametrize("source_dtype", CONVERSION_DTYPES)
@pytest.mark.parametrize("result_dtype", CONVERSION_DTYPES)
def test_materialize_conversion_dtype_pairs(source_dtype, result_dtype):
    with jax.enable_x64():
        values = np.arange(12) % 7
        if source_dtype.startswith("complex"):
            values = values + 1j * (values + 1)
        source = jnp.asarray(values, dtype=source_dtype)
        view = StridedView(source, (2, 3), (1, -3), 9)
        selected = source[jnp.asarray([[9, 6, 3], [10, 7, 4]])]
        if jnp.iscomplexobj(selected) and result_dtype not in ("bool", "complex64", "complex128"):
            selected = selected.real
        expected = selected.astype(result_dtype)
        for execute in (materialize, jax.jit(materialize, static_argnames="dtype")):
            actual = execute(view, dtype=result_dtype)
            assert actual.dtype == jnp.dtype(result_dtype)
            np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3), (0,), (2, 0, 3)])
@pytest.mark.parametrize("shape,strides,offset", [((2, 3), (0, -1), 4), ((), (), 1), ((0,), (1,), 6)])
def test_materialize_conversion_batch_broadcast_scalar_and_empty(batch_shape, shape, strides, offset):
    source = jnp.broadcast_to(jnp.arange(6, dtype=jnp.float32) + .5, (*batch_shape, 6))
    view = StridedView(source, shape, strides, offset)
    indices = np.asarray([
        offset + sum(index * stride for index, stride in zip(coordinate, strides, strict=True))
        for coordinate in np.ndindex(shape)
    ], dtype=np.int64).reshape(shape)
    expected = np.asarray(source)[..., indices].astype(np.int16)
    actual = jax.jit(lambda value: materialize(value, dtype="int16"))(view)
    assert actual.shape == (*batch_shape, *shape)
    np.testing.assert_array_equal(actual, expected)



def test_materialize_conversion_float_integer_boundaries():
    source = jnp.asarray([np.nan, np.inf, -np.inf, -1.9, 1.9, 128, -129], dtype=jnp.float32)
    view = StridedView(source, (7,), (1,), 0)
    np.testing.assert_array_equal(materialize(view, dtype="int8"), [0, 127, -128, -1, 1, 127, -128])
    np.testing.assert_array_equal(materialize(view, dtype="uint8"), [0, 255, 0, 0, 1, 128, 0])



def test_materialize_conversion_integer_narrowing_and_complex_projection():
    integers = StridedView(jnp.asarray([255, 256, 257, -1], dtype=jnp.int32), (4,), (1,), 0)
    np.testing.assert_array_equal(materialize(integers, dtype="uint8"), [255, 0, 1, 255])
    source = jnp.asarray([complex(0, 2), complex(3, np.nan), complex(np.inf, 7)], dtype=jnp.complex64)
    view = StridedView(source, (3,), (1,), 0)
    np.testing.assert_array_equal(materialize(view, dtype="float32"), [0, 3, np.inf])
    np.testing.assert_array_equal(materialize(view, dtype="bool"), [True, True, True])



@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_materialize_conversion_low_precision_rounding(dtype):
    source = jnp.asarray([0., -0., 1.00048828125, 1.00390625, 65520., np.inf, -np.inf, np.nan])
    view = StridedView(source, (8,), (1,), 0)
    actual = np.asarray(materialize(view, dtype=dtype))
    expected = np.asarray(source.astype(dtype))
    np.testing.assert_array_equal(actual[:-1].view(np.uint16), expected[:-1].view(np.uint16))
    assert np.isnan(actual[-1])



@pytest.mark.parametrize("x64", [False, True])
def test_materialize_conversion_output_dtype_resolution(x64):
    with jax.enable_x64(x64):
        view = StridedView(jnp.arange(3, dtype=jnp.int32), (3,), (1,), 0)
        actual = materialize(view, dtype="float64")
        assert actual.dtype == jnp.dtype("float64" if x64 else "float32")



def test_materialize_conversion_partial_multirecord_conversion_and_zero_initialization():
    layout = encode_layout((
        AffineRecord((2,), (-1,), 2, (1,), 1),
        AffineRecord((1,), (1,), 0, (1,), 4),
    ), source_size=3, output_size=6)
    actual = execute_copy(jnp.asarray([5.5, 6.5, 7.5]), layout=layout, output_size=6, dtype="int16")
    np.testing.assert_array_equal(actual, [0, 7, 6, 0, 5, 0])



def test_materialize_conversion_external_vmap_and_target():
    execute = jax.jit(jax.vmap(lambda row: materialize(StridedView(row, (3,), (-2,), 8), dtype="int16")))
    source = jnp.arange(24, dtype=jnp.float32).reshape(2, 12) + .5
    np.testing.assert_array_equal(execute(source), np.asarray(source)[:, [8, 6, 4]].astype(np.int16))
    assert "tensor0_stride_copy_s16_cpu_v1" in execute.lower(source).as_text()



def test_materialize_conversion_unsupported_dtype():
    view = StridedView(jnp.ones(3, dtype=jnp.float32), (3,), (1,), 0)
    with pytest.raises(NotImplementedError, match="does not support"):
        materialize(view, dtype=jnp.float8_e4m3fn)
    unsupported = StridedView(jnp.ones(3, dtype=jnp.float8_e4m3fn), (3,), (1,), 0)
    with pytest.raises(Exception, match="unsupported scalar dtype"):
        materialize(unsupported, dtype="float32").block_until_ready()
