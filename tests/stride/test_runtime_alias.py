from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np
import pytest

from tensor0 import _native
import tensor0._stride._native as native_module
from tensor0._stride._testing import (
    _native_alias_pointers_for_tests,
    _reset_native_alias_pointers_for_tests,
    native_available,
)
from tensor0._stride._plan import (
    CompleteMode,
    AffineRecord,
    StridedOutputInit,
    StridedScalarKind,
    StridedWriteKind,
    build_affine_plan,
)
from tensor0._stride._ops._selected_scale import _build_strided_scale_plan, _selected_scale_alias, _selected_scale_native_descriptor, build_selected_scale_plan
from tests.stride._fixtures import execute_update_scale


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="the Tensor0 extension was built without JAX FFI headers",
)


@lru_cache(maxsize=1)
def _alias_operation_mismatch_target() -> str:
    native_module._ensure_selected_scale_alias_registered("float32")
    registration = _native._stride_prepared_registration()
    target = "tensor0_stride_selected_scale_alias_operation_mismatch_for_tests"
    jax.ffi.register_ffi_target(
        target,
        {
                "instantiate": registration["instantiate"],
            "execute": registration["execute_selected_scale_alias_f32"],
        },
        platform="cpu",
        api_version=1,
    )
    return target


def _call_alias_target(
    target: str,
    base: jax.Array,
    factor: jax.Array,
    descriptor: bytes,
) -> jax.Array:
    layout = tuple(range(base.ndim))
    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct(base.shape, base.dtype),
        input_layouts=(layout, (0,)),
        output_layouts=layout,
        input_output_aliases={0: 0},
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()
    return call(base, factor, descriptor=attribute)


def _multi_record_selected_scale_plan(
    dtype: DTypeLike = jnp.float32,
):
    static = build_affine_plan(
        records=(
            AffineRecord((3,), (2,), 0, (2,), 0),
            AffineRecord((2,), (2,), 1, (2,), 1),
        ),
        output_size=5,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=5,
        source_dtype=dtype,
        result_dtype=dtype,
        output_init=StridedOutputInit.PRESERVE_BASE,
        write_kind=StridedWriteKind.ASSIGN,
        scalar_kind=StridedScalarKind.STATIC_SCALE_CAST,
    )
    return build_selected_scale_plan(static)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_alias_selected_scale_primitive_covers_multi_record_batch_and_vmap(
    dtype: DTypeLike,
) -> None:
    plan = _multi_record_selected_scale_plan(dtype)
    base = jnp.arange(20, dtype=jnp.float32).reshape(4, 5).astype(dtype)
    if jnp.dtype(dtype) == jnp.dtype(jnp.complex64):
        base = base + 1j * (base / 7)
        factors = jnp.asarray(
            [1.25 - 0.5j, -0.75 + 0.25j, 0.5 + 1j, -1 - 0.5j],
            dtype=dtype,
        )
    else:
        factors = jnp.asarray([1.25, -0.75, 0.5, -1], dtype=dtype)
    execute_alias = lambda old, factor: _selected_scale_alias(
        old,
        factor,
        plan=plan,
    )
    execute_ordinary = lambda old, factor: execute_update_scale(
        old,
        factor,
        plan=plan,
    )

    actual = jax.jit(jax.vmap(execute_alias))(base, factors)
    expected = jax.jit(jax.vmap(execute_ordinary))(base, factors)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_alias_selected_scale_primitive_matches_jvp_vjp_and_nested_ad() -> None:
    plan = _build_strided_scale_plan((5,), (2,), 1, 12, "float32")
    base = jnp.linspace(-2, 3, 24, dtype=jnp.float32).reshape(2, 12)
    factor = jnp.asarray([1.25, -0.75], dtype=jnp.float32)
    base_tangent = jnp.linspace(1, -1, 24, dtype=jnp.float32).reshape(2, 12)
    factor_tangent = jnp.asarray([-0.5, 0.25], dtype=jnp.float32)
    cotangent = jnp.linspace(-3, 2, 24, dtype=jnp.float32).reshape(2, 12)
    alias = lambda old, value: _selected_scale_alias(
        old,
        value,
        plan=plan,
    )
    ordinary = lambda old, value: execute_update_scale(old, value, plan=plan)

    alias_jvp = jax.jvp(alias, (base, factor), (base_tangent, factor_tangent))
    ordinary_jvp = jax.jvp(
        ordinary,
        (base, factor),
        (base_tangent, factor_tangent),
    )
    for actual, expected in zip(alias_jvp, ordinary_jvp, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    _, alias_pullback = jax.vjp(alias, base, factor)
    _, ordinary_pullback = jax.vjp(ordinary, base, factor)
    for actual, expected in zip(
        alias_pullback(cotangent),
        ordinary_pullback(cotangent),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    alias_base_pullback = lambda value: jax.vjp(
        lambda old: alias(old, factor),
        base,
    )[1](value)[0]
    ordinary_base_pullback = lambda value: jax.vjp(
        lambda old: ordinary(old, factor),
        base,
    )[1](value)[0]
    alias_nested = jax.jvp(
        alias_base_pullback,
        (cotangent,),
        (jnp.ones_like(cotangent),),
    )
    ordinary_nested = jax.jvp(
        ordinary_base_pullback,
        (cotangent,),
        (jnp.ones_like(cotangent),),
    )
    for actual, expected in zip(alias_nested, ordinary_nested, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_alias_selected_scale_primitive_donation_reuses_batched_base() -> None:
    plan = _build_strided_scale_plan((5,), (2,), 1, 12, "float32")
    execute = jax.jit(
        lambda base, factor: _selected_scale_alias(
            base,
            factor,
            plan=plan,
        ),
        donate_argnums=(0,),
    )
    base = jnp.arange(36, dtype=jnp.float32).reshape(3, 12)
    factor = jnp.asarray([2, -1, 0.5], dtype=jnp.float32)
    expected = execute_update_scale(base, factor, plan=plan)
    expected.block_until_ready()
    pointer = base.unsafe_buffer_pointer()
    compiled = execute.lower(base, factor).compile()

    _reset_native_alias_pointers_for_tests()
    actual = compiled(base, factor)
    actual.block_until_ready()

    native_base, _, native_factor, native_result = (
        _native_alias_pointers_for_tests()
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert base.is_deleted()
    assert native_factor == factor.unsafe_buffer_pointer()
    assert pointer == native_base == native_result == actual.unsafe_buffer_pointer()
    memory = compiled.memory_analysis()
    assert memory is not None
    assert memory.alias_size_in_bytes == actual.nbytes
    assert memory.temp_size_in_bytes == 0


def test_alias_selected_scale_primitive_matches_special_complex_values() -> None:
    plan = _build_strided_scale_plan((9,), (1,), 0, 9, "complex64")
    base = jnp.asarray(
        [
            0 + 0j,
            -0.0 + 0j,
            complex(np.nextafter(np.float32(0), np.float32(1)), 0),
            complex(np.inf, 1),
            complex(-np.inf, -2),
            complex(np.nan, 3),
            complex(3e38, -3e38),
            complex(1.5, -2.25),
            complex(-7, 0.5),
        ],
        dtype=jnp.complex64,
    )
    factors = (
        jnp.asarray(np.nan + 1j, dtype=jnp.complex64),
        jnp.asarray(3e38 + 3e38j, dtype=jnp.complex64),
        jnp.asarray(1.25 - 0.75j, dtype=jnp.complex64),
    )
    for factor in factors:
        actual = jax.jit(
            lambda value: _selected_scale_alias(
                value,
                factor,
                plan=plan,
            )
        )(base)
        expected = jax.jit(
            lambda value: execute_update_scale(value, factor, plan=plan)
        )(base)
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint32),
            np.asarray(expected).view(np.uint32),
        )


def test_alias_selected_scale_primitive_concurrent_calls_are_independent() -> None:
    plan = _build_strided_scale_plan((32,), (2,), 0, 64, "float32")
    execute = jax.jit(
        lambda base, factor: _selected_scale_alias(
            base,
            factor,
            plan=plan,
        )
    )

    def run(index: int) -> np.ndarray:
        base = jnp.arange(64, dtype=jnp.float32) + index
        factor = jnp.asarray(index + 1, dtype=jnp.float32)
        return np.asarray(execute(base, factor))

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(run, range(8)))
    for index, value in enumerate(actual):
        base = jnp.arange(64, dtype=jnp.float32) + index
        expected = execute_update_scale(
            base,
            jnp.asarray(index + 1, dtype=jnp.float32),
            plan=plan,
        )
        np.testing.assert_array_equal(value, np.asarray(expected))


def test_alias_selected_scale_target_requires_pointer_alias() -> None:
    plan = _build_strided_scale_plan((8,), (1,), 0, 8, "float32")
    descriptor = _selected_scale_native_descriptor(plan)
    assert descriptor is not None
    target = native_module._ensure_selected_scale_alias_registered("float32")
    call = jax.ffi.ffi_call(
        target,
        jax.ShapeDtypeStruct((8,), jnp.float32),
        input_layouts=((0,), (0,)),
        output_layouts=(0,),
        vmap_method="expand_dims",
        custom_call_api_version=4,
    )
    base = jnp.arange(8, dtype=jnp.float32)
    factor = jnp.asarray([2], dtype=jnp.float32)
    attribute = np.frombuffer(descriptor, dtype=np.uint8).copy()

    with pytest.raises(Exception, match="requires base/result pointer equality"):
        call(base, factor, descriptor=attribute).block_until_ready()


def test_alias_selected_scale_target_rejects_mutated_descriptor() -> None:
    plan = _build_strided_scale_plan((8,), (1,), 0, 8, "float32")
    descriptor = _selected_scale_native_descriptor(plan)
    assert descriptor is not None
    mutated = bytearray(descriptor)
    mutated[0] ^= 0xFF
    target = native_module._ensure_selected_scale_alias_registered("float32")

    with pytest.raises(Exception, match="descriptor magic mismatch"):
        _call_alias_target(
            target,
            jnp.arange(8, dtype=jnp.float32),
            jnp.asarray([2], dtype=jnp.float32),
            bytes(mutated),
        ).block_until_ready()


def test_alias_selected_scale_target_rejects_descriptor_dtype_mismatch() -> None:
    plan = _build_strided_scale_plan((8,), (1,), 0, 8, "float32")
    descriptor = _selected_scale_native_descriptor(plan)
    assert descriptor is not None
    target = native_module._ensure_selected_scale_alias_registered("complex64")

    with pytest.raises(Exception, match="descriptor dtype does not match"):
        _call_alias_target(
            target,
            jnp.arange(8, dtype=jnp.float32).astype(jnp.complex64),
            jnp.asarray([2], dtype=jnp.complex64),
            descriptor,
        ).block_until_ready()


def test_alias_selected_scale_target_rejects_storage_size_mismatch() -> None:
    plan = _build_strided_scale_plan((8,), (1,), 0, 8, "float32")
    descriptor = _selected_scale_native_descriptor(plan)
    assert descriptor is not None
    target = native_module._ensure_selected_scale_alias_registered("float32")

    with pytest.raises(Exception, match="storage size does not match descriptor"):
        _call_alias_target(
            target,
            jnp.arange(9, dtype=jnp.float32),
            jnp.asarray([2], dtype=jnp.float32),
            descriptor,
        ).block_until_ready()


def test_alias_selected_scale_target_rejects_prepared_operation_mismatch() -> None:
    plan = _build_strided_scale_plan((8,), (1,), 0, 8, "float32")
    descriptor = _selected_scale_native_descriptor(plan)
    assert descriptor is not None

    with pytest.raises(Exception, match="operation does not match FFI target"):
        _call_alias_target(
            _alias_operation_mismatch_target(),
            jnp.arange(8, dtype=jnp.float32),
            jnp.asarray([2], dtype=jnp.float32),
            descriptor,
        ).block_until_ready()
