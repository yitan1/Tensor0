from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0 import _native
from tensor0._stride import (
    StridedView,
    disable_threads,
    dotu,
    enable_threads,
    get_num_threads,
    materialize,
    reduce_sum,
    set_num_threads,
    scale,
)
from ._support import native_available


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="native CPU stride unavailable",
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

        set_num_threads((1 << 64) - 2)
        assert get_num_threads() == (1 << 64) - 2
    finally:
        _restore_threads(original)


@pytest.mark.parametrize("value", (True, False, 1.5, "4", None))
def test_set_num_threads_rejects_non_integer(value: object) -> None:
    original = get_num_threads()
    try:
        set_num_threads(2)
        with pytest.raises(TypeError, match="positive integer"):
            set_num_threads(value)  # type: ignore[arg-type]
        assert get_num_threads() == 2
    finally:
        _restore_threads(original)


@pytest.mark.parametrize("value", (0, -1, (1 << 64) - 1, 1 << 64))
def test_set_num_threads_rejects_out_of_range_integer(value: int) -> None:
    original = get_num_threads()
    try:
        set_num_threads(2)
        with pytest.raises(ValueError):
            set_num_threads(value)
        assert get_num_threads() == 2
    finally:
        _restore_threads(original)


@pytest.mark.parametrize("operation", [get_num_threads, disable_threads, enable_threads,
                                       lambda: set_num_threads(2)])
def test_thread_controls_reject_unavailable_backend(operation, monkeypatch) -> None:
    monkeypatch.setattr(_native, "_stride_ffi_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        operation()


@pytest.mark.parametrize("batch_count", [1, 7])
def test_compiled_executable_reused_after_thread_limit_changes(batch_count) -> None:
    size = 1 << 17
    source = (jnp.arange(batch_count * size, dtype=jnp.float32) % 4).reshape(batch_count, size)
    traces = []

    @jax.jit
    def operation(value):
        traces.append(1)
        view = StridedView(value, (size,), (1,), 0)
        return materialize(view), scale(view, 0.75).data, reduce_sum(view), dotu(view, view)

    executable = operation.lower(source).compile()
    original = get_num_threads()
    try:
        set_num_threads(1)
        sequential = jax.block_until_ready(executable(source))

        set_num_threads(4)
        parallel = jax.block_until_ready(executable(source))
        enable_threads()
        unrestricted = jax.block_until_ready(executable(source))
        set_num_threads(1)
        sequential_again = jax.block_until_ready(executable(source))
    finally:
        _restore_threads(original)

    assert traces == [1]
    host = np.asarray(source)
    expected = (host, host * np.float32(0.75),
                host.sum(axis=-1, dtype=np.float64),
                np.square(host.astype(np.float64)).sum(axis=-1))
    for results in (sequential, parallel, unrestricted, sequential_again):
        for actual, reference in zip(results, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=0)
        for index in (0, 1):
            assert np.asarray(results[index]).tobytes() == np.asarray(sequential[index]).tobytes()
