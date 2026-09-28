"""Production execution helpers separate from independent oracles."""

from contextlib import contextmanager
from functools import cache

import jax.numpy as jnp

from tensor0._stride import enable_threads, get_num_threads, set_num_threads
from tensor0._stride._jax import accumulation_p, reduction_p, update_p

from tests.stride.support.oracles.reduction import (
    AD_RECORDS, AXES, COEFFICIENT_RECORDS, OUTPUT_SHAPES, RECORDS, reference_functions,
)
from tests.stride.support.oracles.update import LAYOUTS, assert_disjoint_writes, update_reference
from tests.stride.support.samples import values


@cache
def reduction_functions(operation, dtype):
    def run(source, first, second):
        if operation == "accumulation":
            return accumulation_p.bind(source, first, second, records=AD_RECORDS,
                coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
        return reduction_p.bind(source, first, second, records=AD_RECORDS,
            output_shapes=((2, 1), (2, 1), (1,)), reduction_axes=((False, True), (False, True), (True,)),
            coefficient_records=(0, 1), output_size=6, dtype=jnp.dtype(dtype))
    return run, reference_functions(operation, dtype)


def execute_reduction(source, empty, first, second):
    return reduction_p.bind(
        source, empty, first, second, records=RECORDS, output_shapes=OUTPUT_SHAPES,
        reduction_axes=AXES, coefficient_records=COEFFICIENT_RECORDS, output_size=7, dtype=source.dtype,
    )


@cache
def update_functions(records):
    assert_disjoint_writes(records)

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=records)

    return execute, update_reference(records)


def mapped(source, record, coefficient, dtype, output_size):
    base = jnp.zeros((*source.shape[:-1], output_size), dtype=dtype)
    return update_p.bind(source, base, coefficient, jnp.int32(0), records=(record,))


@contextmanager
def thread_limit(limit):
    previous = get_num_threads()
    set_num_threads(limit)
    try:
        yield
    finally:
        if previous is None:
            enable_threads()
        else:
            set_num_threads(previous)


def shared_coefficient_case(source_dtype, coefficient_dtype, dtype, operation):
    first = jnp.asarray(2, dtype=coefficient_dtype)
    second = jnp.asarray(0.5, dtype=dtype)
    if operation == 'update':
        run, reference = update_functions(LAYOUTS[1])
        arguments = (values((6,), source_dtype), values((10,), dtype), first, second)
    else:
        run, reference = reduction_functions(operation, dtype)
        arguments = (values((5,), source_dtype), first, second)
    return run, reference, arguments
