from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import (
    AffineRecord,
    CompleteMode,
    build_affine_plan,
)
from tensor0._stride import (
    disable_threads,
    enable_threads,
    get_num_threads,
    set_num_threads,
)
from tensor0._stride._map import _execute_map
from tensor0._stride._native import native_available
from tensor0._stride._testing import _native_worker_counts_for_tests


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="native CPU stride unavailable",
)


def _scaled_plan(size: int):
    return build_affine_plan(
        records=(AffineRecord((size,), (1,), 0, (1,), 0, 0.75),),
        output_size=size,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=size,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )


def _restore_threads(limit: int | None) -> None:
    if limit is None:
        enable_threads()
    else:
        set_num_threads(limit)


def test_thread_configuration_roundtrip() -> None:
    original = get_num_threads()
    try:
        set_num_threads(4)
        assert get_num_threads() == 4

        disable_threads()
        assert get_num_threads() == 1

        enable_threads()
        assert get_num_threads() is None
    finally:
        _restore_threads(original)


@pytest.mark.parametrize("value", (True, False, 1.5, "4", None))
def test_set_num_threads_rejects_non_integer(value: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        set_num_threads(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (0, -1, (1 << 64) - 1, 1 << 64))
def test_set_num_threads_rejects_out_of_range_integer(value: int) -> None:
    with pytest.raises(ValueError):
        set_num_threads(value)


def test_compiled_executable_observes_runtime_thread_limit() -> None:
    plan = _scaled_plan(1 << 20)
    source = jnp.arange(plan.source_size, dtype=jnp.float32)
    executable = jax.jit(
        lambda value: _execute_map(
            value,
            plan=plan,
        )
    ).lower(source).compile()
    original = get_num_threads()
    try:
        set_num_threads(1)
        sequential = executable(source)
        sequential.block_until_ready()
        sequential_workers, available = _native_worker_counts_for_tests()

        set_num_threads(4)
        parallel = executable(source)
        parallel.block_until_ready()
        parallel_workers, parallel_available = _native_worker_counts_for_tests()

        set_num_threads(available + 17)
        unrestricted = executable(source)
        unrestricted.block_until_ready()
        unrestricted_workers, unrestricted_available = (
            _native_worker_counts_for_tests()
        )
    finally:
        _restore_threads(original)

    assert sequential_workers == 1
    assert parallel_available == available
    assert parallel_workers == min(available, 4)
    assert unrestricted_available == available
    assert unrestricted_workers <= max(available, 1)
    assert np.asarray(parallel).tobytes() == np.asarray(sequential).tobytes()
    assert np.asarray(unrestricted).tobytes() == np.asarray(sequential).tobytes()
