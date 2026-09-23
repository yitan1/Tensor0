"""JAX donation, live inputs and executable reuse."""

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import StridedView, scale
from tensor0._stride._ffi._registration import operation_target
from tensor0._stride._jax import update_p

from tests.stride.support.availability import native_available
from tests.stride.support.layouts import PARTIAL
from tests.stride.support.oracles.raw_update import reference_update
from tests.stride.support.oracles.scale import reference_scale


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
def test_donation_reuses_batched_scale_base_buffer():
    base = jnp.arange(36, dtype=jnp.float32).reshape(3, 12)
    factor = jnp.asarray([2, -1, .5], dtype=jnp.float32)
    expected = np.asarray(base).copy()
    expected[:, 1:11:2] *= np.asarray(factor)[:, None]
    original_factor = np.asarray(factor).copy()
    pointer = base.unsafe_buffer_pointer()
    compiled = jax.jit(lambda old, value: scale(StridedView(old, (5,), (2,), 1), value).data,
                       donate_argnums=(0,)).lower(base, factor).compile()
    actual = compiled(base, factor)
    actual.block_until_ready()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(factor, original_factor)
    assert base.is_deleted()
    assert actual.unsafe_buffer_pointer() == pointer
    memory = compiled.memory_analysis()
    assert memory is not None and memory.alias_size_in_bytes == actual.nbytes


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("factor", [complex(np.nan, 1), complex(3e38, 3e38), 1.25 - .75j])
def test_donated_and_preserved_inputs_share_complex_arithmetic(factor):
    values = jnp.asarray([
        0j, complex(-0., 0.), complex(np.nextafter(np.float32(0), np.float32(1)), 0),
        complex(np.inf, 1), complex(-np.inf, -2), complex(np.nan, 3),
        complex(3e38, -3e38), 1.5 - 2.25j, -7 + .5j,
    ], dtype=jnp.complex64)
    coefficient = jnp.asarray(factor, dtype=jnp.complex64)
    operation = lambda old, value: scale(StridedView(old, (9,), (1,), 0), value).data
    expected = jax.jit(operation)(values, coefficient)
    donated = jnp.array(values, copy=True)
    actual = jax.jit(operation, donate_argnums=(0,))(donated, coefficient)
    actual.block_until_ready()
    assert donated.is_deleted()
    for result, wanted in ((actual.real, expected.real), (actual.imag, expected.imag)):
        np.testing.assert_allclose(result, wanted, rtol=2e-6, atol=1e-6, equal_nan=True)


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("donate", [False, True])
def test_concurrent_calls_to_one_prepared_executable_are_independent(donate):
    operation = jax.jit(lambda old, value: scale(StridedView(old, (32,), (2,), 0), value).data,
                        donate_argnums=(0,) if donate else ())
    compiled = operation.lower(jax.ShapeDtypeStruct((64,), jnp.float32),
                               jax.ShapeDtypeStruct((), jnp.float32)).compile()

    def run(index):
        host = np.arange(64, dtype=np.float32) + index
        base = jnp.asarray(host)
        result = np.asarray(compiled(base, jnp.float32(index + 1)))
        if donate:
            assert base.is_deleted()
        else:
            np.testing.assert_array_equal(base, host)
        expected = host.copy()
        expected[::2] *= index + 1
        np.testing.assert_array_equal(result, expected)
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(8)))
    assert len(results) == 8


@pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')
@pytest.mark.parametrize("donate", [False, True])
def test_selected_scale_dynamic_operand_and_input_reuse(donate):
    operation = jax.jit(lambda old, factor: scale(StridedView(old, (16,), (3,), 2), factor).data,
                        donate_argnums=(0,) if donate else ())
    template = jnp.arange(50, dtype=jnp.float32)
    lowered = operation.lower(template, jnp.float32(2))
    text = lowered.as_text()
    assert text.count("custom_call") == 1 and "output_operand_aliases" in text
    assert operation_target("update", np.dtype(jnp.float32)) in text
    executable = lowered.compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.alias_size_in_bytes == (template.nbytes if donate else 0)
    for coefficient in (jnp.float32(2), jnp.float32(-3)):
        base = jnp.array(template, copy=True)
        expected = reference_scale(template, coefficient)
        np.testing.assert_array_equal(executable(base, coefficient), expected)
        if donate:
            assert base.is_deleted()
        else:
            np.testing.assert_array_equal(base, template)


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("donate", [False, True])
def test_update_base_reuse_respects_functional_input_protection(beta, donate):
    source, base = jnp.arange(32, dtype=jnp.float32), jnp.arange(50, dtype=jnp.float32)
    expected = reference_update(source, base, PARTIAL, .5, beta)
    operation = jax.jit(lambda old, new: update_p.bind(new, old, jnp.float32(.5), jnp.int32(beta), records=PARTIAL),
                        donate_argnums=(0,) if donate else ())
    lowered = operation.lower(base, source)
    text = lowered.as_text()
    assert "output_operand_aliases" in text and "operand_index = 1" in text
    assert "tensor0_stride_update_f32_cpu_v1" in text
    executable = lowered.compile()
    memory = executable.memory_analysis()
    assert memory is not None
    assert memory.alias_size_in_bytes == (base.nbytes if donate else 0)
    actual = executable(base, source)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, np.arange(32))
    if donate:
        assert base.is_deleted()
    else:
        np.testing.assert_array_equal(base, np.arange(50))
