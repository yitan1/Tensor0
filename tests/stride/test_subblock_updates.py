from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import SU2Irrep, hom, space
from tensor0._stride import StridedView, scale
from tests.stride._fixtures import execute_view_accumulate, execute_view_assign
from tensor0.structure import get_degeneracystructure

from ._oracle import (
    legacy_strided_accumulate,
    legacy_strided_assign,
    legacy_strided_indices,
    legacy_strided_scale,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _noncompact_subblock_case():
    factor = space(SU2Irrep, {0: 2, 1: 3})
    layout = get_degeneracystructure(hom((factor, factor), (factor, factor)))
    return layout, layout.subblockstructure[0]


def _view(data, metadata: dict[str, object]) -> StridedView:
    return StridedView(
        data,
        metadata["sizes"],  # type: ignore[arg-type]
        metadata["strides"],  # type: ignore[arg-type]
        metadata["offset"],  # type: ignore[arg-type]
    )


def _source_view(data, sizes: tuple[int, ...]) -> StridedView:
    return StridedView.from_dense(data, sizes)


def test_static_subblock_metadata_updates_match_bounded_oracle() -> None:
    layout, subblock = _noncompact_subblock_case()
    sizes = tuple(subblock.sizes)
    strides = tuple(subblock.strides)
    metadata = {"sizes": sizes, "strides": strides, "offset": subblock.offset}
    storage = jnp.linspace(-2, 3, layout.total_dim, dtype=jnp.float32)
    value = jnp.linspace(4, -1, int(np.prod(sizes)), dtype=jnp.float32).reshape(sizes)
    factor = jnp.asarray(-1.25, dtype=jnp.float32)
    expected_assign = legacy_strided_assign(
        storage,
        sizes,
        strides,
        subblock.offset,
        value,
    )
    expected_accumulate = legacy_strided_accumulate(
        storage,
        sizes,
        strides,
        subblock.offset,
        value,
    )
    expected_scale = legacy_strided_scale(
        storage,
        sizes,
        strides,
        subblock.offset,
        factor,
    )

    destination = _view(storage, metadata)
    source = _source_view(value, sizes)
    actual_assign = execute_view_assign(destination, source)
    actual_accumulate = execute_view_accumulate(destination, source)
    actual_scale = scale(destination, factor).data

    np.testing.assert_array_equal(actual_assign, expected_assign)
    np.testing.assert_array_equal(actual_accumulate, expected_accumulate)
    np.testing.assert_array_equal(actual_scale, expected_scale)


def test_strided_updates_reject_non_jax_array_operands() -> None:
    base = jnp.arange(8, dtype=jnp.float32)
    numpy_base = np.arange(8, dtype=np.float32)
    numpy_source = np.asarray([10, 20], dtype=np.float32)
    metadata = {"sizes": (2,), "strides": (2,), "offset": 0}

    with pytest.raises(TypeError, match="data must be a JAX Array"):
        _view(numpy_base, metadata)
    destination = _view(base, metadata)
    for operation in (execute_view_assign, execute_view_accumulate):
        with pytest.raises(TypeError, match="source must be a StridedView"):
            operation(destination, numpy_source)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="view must be a StridedView"):
        scale(numpy_base, 2).data  # type: ignore[arg-type]


def test_static_subblock_update_jit_vmap_and_ad_match_compatibility() -> None:
    layout, subblock = _noncompact_subblock_case()
    sizes = tuple(subblock.sizes)
    strides = tuple(subblock.strides)
    metadata = {"sizes": sizes, "strides": strides, "offset": subblock.offset}
    storage = jnp.linspace(-2, 3, layout.total_dim, dtype=jnp.float32)
    value = jnp.linspace(4, -1, int(np.prod(sizes)), dtype=jnp.float32).reshape(sizes)
    factor = jnp.asarray(-1.25, dtype=jnp.float32)
    storage_tangent = jnp.linspace(1, -2, layout.total_dim, dtype=jnp.float32)
    factor_tangent = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jnp.linspace(-3, 4, layout.total_dim, dtype=jnp.float32)
    migrated = lambda old, coefficient: scale(_view(old, metadata), coefficient).data
    indices = legacy_strided_indices(sizes, strides, subblock.offset)
    legacy = lambda old, scale: old.at[indices].set(
        scale * old[indices],
        unique_indices=True,
    )

    np.testing.assert_array_equal(
        jax.jit(
            lambda old, update: execute_view_assign(
                _view(old, metadata),
                _source_view(update, sizes),
            )
        )(
            storage,
            value,
        ),
        legacy_strided_assign(storage, sizes, strides, subblock.offset, value),
    )
    np.testing.assert_array_equal(
        jax.jit(
            lambda old, update: execute_view_accumulate(
                _view(old, metadata),
                _source_view(update, sizes),
            )
        )(
            storage,
            value,
        ),
        legacy_strided_accumulate(storage, sizes, strides, subblock.offset, value),
    )
    actual_primal, actual_tangent = jax.jvp(
        migrated,
        (storage, factor),
        (storage_tangent, factor_tangent),
    )
    expected_primal, expected_tangent = jax.jvp(
        legacy,
        (storage, factor),
        (storage_tangent, factor_tangent),
    )
    np.testing.assert_array_equal(actual_primal, expected_primal)
    np.testing.assert_allclose(
        actual_tangent,
        expected_tangent,
        rtol=1e-6,
        atol=1e-6,
    )
    actual_base, actual_factor = jax.vjp(migrated, storage, factor)[1](cotangent)
    expected_base, expected_factor = jax.vjp(legacy, storage, factor)[1](cotangent)
    np.testing.assert_array_equal(actual_base, expected_base)
    # The native compact map changes the reduction tree relative to the legacy
    # gather.  Require one-ULP agreement without rebuilding O(N) addresses.
    np.testing.assert_array_max_ulp(actual_factor, expected_factor, maxulp=1)

    storage_batch = jnp.stack((storage, 2 * storage, -storage))
    factor_batch = jnp.asarray([2, -0.5, 3], dtype=jnp.float32)
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(migrated))(storage_batch, factor_batch),
        jax.jit(jax.vmap(legacy))(storage_batch, factor_batch),
    )


def test_static_subblock_assign_preserves_complex_special_values() -> (
    None
):
    layout, subblock = _noncompact_subblock_case()
    sizes = tuple(subblock.sizes)
    strides = tuple(subblock.strides)
    storage = jnp.full(
        layout.total_dim,
        complex(-0.0, 0.0),
        dtype=jnp.complex64,
    )
    pattern = jnp.asarray(
        [
            complex(jnp.inf, 0.0),
            complex(0.0, jnp.inf),
            complex(-jnp.inf, 2.0),
            complex(jnp.nan, -3.0),
        ],
        dtype=jnp.complex64,
    )
    value = jnp.resize(pattern, sizes)
    metadata = {"sizes": sizes, "strides": strides, "offset": subblock.offset}

    destination = _view(storage, metadata)
    source = _source_view(value, sizes)
    actual_assign = execute_view_assign(destination, source)
    expected_assign = legacy_strided_assign(
        storage,
        sizes,
        strides,
        subblock.offset,
        value,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_assign).view(np.uint32),
        np.asarray(expected_assign).view(np.uint32),
    )


def test_concrete_eager_stride_operations_reject_packed_storage_sharding() -> None:
    script = textwrap.dedent(
        """
        import json

        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        from tensor0._stride import (
            StridedView,
            materialize,
            scale,
        )
        from tests.stride._fixtures import execute_view_accumulate, execute_view_assign

        mesh = Mesh(np.asarray(jax.devices()), ("device",))
        storage_sharding = NamedSharding(mesh, P("device"))
        replicated = NamedSharding(mesh, P())
        storage = jax.device_put(
            np.arange(8, dtype=np.float32),
            storage_sharding,
        )
        update = jax.device_put(
            np.asarray([10, 20], dtype=np.float32),
            replicated,
        )
        factor = jax.device_put(np.asarray(2, dtype=np.float32), replicated)
        operations = {
            "materialize": lambda: materialize(
                StridedView(storage, (2,), (2,), 0),
            ),
            "assign": lambda: execute_view_assign(
                StridedView(storage, (2,), (2,), 0),
                StridedView.from_dense(update, (2,)),
            ),
            "accumulate": lambda: execute_view_accumulate(
                StridedView(storage, (2,), (2,), 0),
                StridedView.from_dense(update, (2,)),
            ),
            "scale": lambda: scale(StridedView(storage, (2,), (2,), 0), factor).data,
        }
        results = {}
        for name, operation in operations.items():
            try:
                operation().block_until_ready()
            except Exception as error:
                results[name] = (
                    "cannot shard the packed storage axis" in str(error)
                )
            else:
                results[name] = False
        print(json.dumps(results))
"""
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "materialize": True,
        "assign": True,
        "accumulate": True,
        "scale": True,
    }
