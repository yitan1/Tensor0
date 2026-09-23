"""Shared coefficient cases for Update, Reduction and accumulation AD checks."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from tests.stride.support.data import PAIRS as MIXED_PAIRS
from tests.stride.support.oracles.reduction import functions as reduction_ad_functions
from tests.stride.support.oracles.update import LAYOUTS, update_functions
from tests.stride.support.samples import values


def shared_coefficient_case(source_dtype, coefficient_dtype, dtype, operation):
    first = jnp.asarray(2, dtype=coefficient_dtype)
    second = jnp.asarray(0.5, dtype=dtype)
    if operation == 'update':
        run, reference = update_functions(LAYOUTS[1])
        arguments = (values((6,), source_dtype), values((10,), dtype), first, second)
    else:
        run, reference = reduction_ad_functions(operation, dtype)
        arguments = (values((5,), source_dtype), first, second)
    return run, reference, arguments


DTYPES = ('float16', 'bfloat16', 'float32', 'float64', 'complex64', 'complex128')


COMBINATIONS = [types for types in product(DTYPES, repeat=3) if any((dtype in ('float16', 'bfloat16') for dtype in types))]


# Forward execution retains every supported three-way dtype combination.
# AD covers semantic representatives; it is not an exhaustive dtype matrix.
SHARED_CASES = [(operation, *types)
                for operation in ('accumulation', 'reduction', 'update')
                for types in COMBINATIONS
                if operation != 'update' or types[0] == types[2]
                or (types[0], types[2]) in MIXED_PAIRS]


DISCRETE = ('bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')


def check(run, arguments):
    expected = run(*arguments)
    primal, tangent = jax.jit(lambda *values: jax.jvp(run, values, tuple((jnp.ones_like(value) for value in values))))(*arguments)
    np.testing.assert_array_equal(primal, expected)
    assert primal.dtype == expected.dtype
    assert tangent.dtype == jax.dtypes.float0 and tangent.shape == primal.shape
    _, pullback = jax.vjp(run, *arguments)
    cotangent = np.zeros(primal.shape, dtype=jax.dtypes.float0)
    for gradient, argument in zip(jax.jit(pullback)(cotangent), arguments, strict=True):
        assert gradient.shape == argument.shape and gradient.dtype == argument.dtype
        np.testing.assert_array_equal(gradient, jnp.zeros_like(argument))
