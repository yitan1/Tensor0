"""Regression checks for coefficient AD harness trace reuse."""

import jax
import jax.numpy as jnp
import pytest

from tests.stride.support.ad import check_coefficient_ad
from tests.stride.support.availability import native_available
from tests.stride.support.oracles.coefficients import shared_coefficient_case


pytestmark = pytest.mark.skipif(not native_available(), reason='native CPU stride is unavailable')


@pytest.fixture(autouse=False)
def enable_x64():
    with jax.enable_x64():
        yield


@pytest.mark.usefixtures("enable_x64")
@pytest.mark.parametrize('operation', ['accumulation', 'reduction', 'update'])
def test_ad_harness_reuses_trace_with_dynamic_primals(operation):
    execute, reference, arguments = shared_coefficient_case('float32', 'float32', 'float32', operation)
    traces = []

    def run(*inputs):
        traces.append(None)
        return execute(*inputs)

    # The same compiled function must reload both data and coefficients, not
    # specialize the derivative to captured buffers or zero/one factor values.
    count = None
    for factor in (0., 1., 2.):
        inputs = tuple(value + factor for value in arguments[:-2]) + (
            jnp.float32(factor), jnp.float32(2 - factor))
        check_coefficient_ad(run, reference, inputs, coefficient_count=2, complex_cotangent=1 + 2j)
        if count is None:
            count = len(traces)
            assert count > 0
        else:
            assert len(traces) == count
