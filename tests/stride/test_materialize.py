from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tensor0._stride._materialize as materialize_module
from tensor0 import SU2Irrep, hom, space
from tensor0._stride import (
    StridedView,
    materialize,
    strided_copy,
)
from tensor0._stride._ffi import (
    _native_call_count_for_tests,
    _reset_native_call_count_for_tests,
    native_available,
)
from tensor0._stride._materialize import build_materialize_plan
from tensor0._stride._routing import (
    RouteKind,
    StableHloCapability,
    compile_affine_route_features,
    decide_fresh_map_route,
)
from tensor0.structure import get_degeneracystructure

from ._oracle import legacy_materialize


def _view(
    data,
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> StridedView:
    return StridedView(data, sizes, strides, offset)


def test_static_contiguous_materialization_uses_address_free_compact_recipe() -> None:
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
    assert "custom_call" not in hlo
    assert "gather" not in hlo
    assert "scatter" not in hlo


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_static_sliced_permuted_materialization_and_cast_use_native() -> None:
    source = jnp.arange(20, dtype=jnp.float32)
    expected = legacy_materialize(source, (2, 3), (1, 4), 2).astype(
        jnp.complex64
    )
    function = jax.jit(
        lambda value: materialize(
            _view(value, (2, 3), (1, 4), 2),
            result_dtype=jnp.complex64,
        )
    )

    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.dtype == jnp.complex64
    assert _native_call_count_for_tests() == 1
    hlo = str(function.lower(source).compiler_ir("stablehlo")).lower()
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
        result_dtype=jnp.complex64,
    )

    assert actual.shape == (2, 0, 3)
    assert actual.dtype == jnp.complex64


def test_materialize_requires_a_bound_view() -> None:
    source = np.arange(20, dtype=np.float32)

    with pytest.raises(TypeError, match="materialize requires a StridedView"):
        materialize(source)  # type: ignore[arg-type]

    plan = build_materialize_plan(
        sizes=(2, 3),
        strides=(1, 4),
        offset=2,
        source_size=source.size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    with pytest.raises(TypeError, match="strided_copy source.*JAX Array"):
        strided_copy(source, plan=plan)


def test_small_concrete_materialization_enters_unified_executor(monkeypatch) -> None:
    source = jnp.arange(20, dtype=jnp.float32)
    real_strided_copy = materialize_module.strided_copy
    calls = 0

    def counted_strided_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_strided_copy(*args, **kwargs)

    monkeypatch.setattr(materialize_module, "strided_copy", counted_strided_copy)
    actual = materialize(
        _view(source, (2, 3), (1, 4), 2),
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(legacy_materialize(source, (2, 3), (1, 4), 2)),
    )
    assert calls == 1


def test_materialize_jit_jvp_vjp_linear_transpose_and_vmap_match_legacy() -> None:
    source = jnp.linspace(-1, 1, 20, dtype=jnp.float32)
    tangent = jnp.linspace(1, 2, 20, dtype=jnp.float32)
    cotangent = jnp.linspace(-2, 1, 6, dtype=jnp.float32).reshape(2, 3)

    def migrated(value):
        return materialize(
            _view(value, (2, 3), (1, 4), 2),
        )

    def legacy(value):
        return legacy_materialize(value, (2, 3), (1, 4), 2)

    migrated_primal, migrated_tangent = jax.jvp(
        migrated,
        (source,),
        (tangent,),
    )
    legacy_primal, legacy_tangent = jax.jvp(
        legacy,
        (source,),
        (tangent,),
    )
    migrated_vjp = jax.vjp(migrated, source)[1](cotangent)[0]
    legacy_vjp = jax.vjp(legacy, source)[1](cotangent)[0]
    migrated_transpose = jax.linear_transpose(
        migrated,
        jnp.zeros_like(source),
    )(cotangent)[0]
    legacy_transpose = jax.linear_transpose(
        legacy,
        jnp.zeros_like(source),
    )(cotangent)[0]
    batch = jnp.stack((source, 2 * source, -source))

    np.testing.assert_array_equal(migrated_primal, legacy_primal)
    np.testing.assert_array_equal(migrated_tangent, legacy_tangent)
    np.testing.assert_array_equal(migrated_vjp, legacy_vjp)
    np.testing.assert_array_equal(migrated_transpose, legacy_transpose)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(batch),
        jax.jit(jax.vmap(legacy))(batch),
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
            result_dtype=jnp.complex64,
        )

    def explicit(value):
        return legacy_materialize(value, (2, 3), (3, 1), 0).astype(
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
    expected = legacy_materialize(
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
def test_partial_materialization_does_not_inherit_complete_map_threshold() -> None:
    sizes = (64, 64)
    plan = build_materialize_plan(
        sizes=sizes,
        strides=(1, 64),
        offset=0,
        source_size=64 * 64 + 1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    features = compile_affine_route_features(
        plan,
        source_shape=(plan.source_size,),
        platform="cpu",
        stablehlo_capability=StableHloCapability.NONE,
        native_runtime_available=True,
    )
    decision = decide_fresh_map_route(features)

    assert not features.source_complete
    assert decision.route is RouteKind.NATIVE
    assert decision.reason == "general_native"


@pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")
def test_large_partial_materialization_obeys_general_native_threshold() -> None:
    sizes = (512, 512)
    strides = (1, 512)
    copied = sizes[0] * sizes[1]
    source = jnp.arange(copied + 1, dtype=jnp.float32)
    plan = build_materialize_plan(
        sizes=sizes,
        strides=strides,
        offset=0,
        source_size=source.size,
        source_dtype=source.dtype,
        result_dtype=source.dtype,
    )
    features = compile_affine_route_features(
        plan,
        source_shape=tuple(source.shape),
        platform="cpu",
        stablehlo_capability=StableHloCapability.NONE,
        native_runtime_available=True,
    )
    decision = decide_fresh_map_route(features)
    function = jax.jit(
        lambda value: materialize(
            _view(value, sizes, strides, 0),
        )
    )

    hlo = str(function.lower(source).compiler_ir("stablehlo")).lower()
    _reset_native_call_count_for_tests()
    actual = function(source)
    actual.block_until_ready()

    assert not features.source_complete
    assert decision.route is RouteKind.NATIVE
    assert decision.reason == "general_native"
    assert hlo.count("custom_call") == 1
    assert "gather" not in hlo
    assert "scatter" not in hlo
    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(source[:-1].reshape(sizes).T),
    )

    _reset_native_call_count_for_tests()
    eager = materialize(
        _view(source, sizes, strides, 0),
    )
    eager.block_until_ready()

    assert _native_call_count_for_tests() == 1
    np.testing.assert_array_equal(eager, actual)
