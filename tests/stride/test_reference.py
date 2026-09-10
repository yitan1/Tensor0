from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._plan import (
    CompleteMode,
    AffineRecord,
    build_affine_plan,
)

from ._fixtures import (
    contiguous_dtype_plan,
    partial_mixed_plan,
    two_record_noncompact_plan,
)
from ._oracle import (
    REFERENCE_LIVE_BYTE_CAP,
    REFERENCE_PYTHON_ADDRESS_BYTE_CAP,
    REFERENCE_RECORD_CAP,
    enumerate_addresses,
    execute_reference,
    reference_ineligibility_reasons,
)


def _numpy_oracle(source: np.ndarray) -> np.ndarray:
    bound = two_record_noncompact_plan()
    result = np.zeros((bound.output_size,), dtype=np.float32)
    for record in bound.records:
        source_addresses = enumerate_addresses(record, "source")
        destination_addresses = enumerate_addresses(record, "destination")
        scale = np.asarray(record.scale, dtype=np.float32)
        result[np.asarray(destination_addresses)] = (
            scale * source[np.asarray(source_addresses)]
        )
    return result


def test_reference_matches_independent_address_oracle() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    actual = execute_reference(source, bound)

    np.testing.assert_allclose(np.asarray(actual), _numpy_oracle(np.asarray(source)))


def test_reference_supports_leading_batch_prefix_under_jit() -> None:
    bound = two_record_noncompact_plan()
    source = jnp.arange(3 * bound.source_size, dtype=jnp.float32).reshape(
        3, bound.source_size
    )

    actual = jax.jit(lambda value: execute_reference(value, bound))(source)
    expected = np.stack(tuple(_numpy_oracle(np.asarray(row)) for row in source))

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_reference_partial_plan_zero_fills_uncovered_destinations() -> None:
    bound = partial_mixed_plan()
    source = jnp.arange(bound.source_size, dtype=jnp.float32)

    actual = np.asarray(execute_reference(source, bound))
    covered = np.arange(2, 50, 3)
    expected = np.zeros((50,), dtype=np.float32)
    expected[covered] = 0.5 * np.arange(1, 32, 2, dtype=np.float32)

    np.testing.assert_array_equal(actual, expected)


def test_oracle_resource_policy_bounds_materialized_addresses() -> None:
    python_size = REFERENCE_PYTHON_ADDRESS_BYTE_CAP // 72 + 1
    python_bound = contiguous_dtype_plan(jnp.float32, 1, size=python_size)
    assert reference_ineligibility_reasons(python_bound) == (
        "reference_python_address_byte_cap",
    )

    small = contiguous_dtype_plan(jnp.float32, 1, size=4_096)
    batch = REFERENCE_LIVE_BYTE_CAP // (4 * small.source_size) + 1
    assert reference_ineligibility_reasons(
        small,
        source_shape=(batch, small.source_size),
    ) == ("reference_live_byte_cap",)

    record_count = REFERENCE_RECORD_CAP + 1
    records = tuple(
        AffineRecord((1,), (1,), index, (1,), index)
        for index in range(record_count)
    )
    too_many_records = build_affine_plan(
        records=records,
        output_size=record_count,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=record_count,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    assert reference_ineligibility_reasons(too_many_records) == (
        "reference_record_cap",
    )

    wide_address = build_affine_plan(
        records=(AffineRecord((1,), (1,), 2**31, (1,), 0),),
        output_size=1,
        coverage=CompleteMode.COMPLETE_UNIQUE,
        source_size=2**31 + 1,
        source_dtype=jnp.float32,
        result_dtype=jnp.float32,
    )
    assert reference_ineligibility_reasons(wide_address) == (
        "reference_index_width",
    )
    with jax.enable_x64():
        assert reference_ineligibility_reasons(wide_address) == ()

    with pytest.raises(RuntimeError, match="reference_python_address_byte_cap"):
        jax.eval_shape(
            lambda value: execute_reference(value, python_bound),
            jax.ShapeDtypeStruct(
                (python_bound.source_size,),
                jnp.float32,
            ),
        )
