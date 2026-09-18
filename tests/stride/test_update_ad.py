"""Update AD for disjoint writes, repeated reads and sequential composition."""

from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord


LAYOUTS = (
    (AffineRecord((3,), (1,), 0, (2,), 1), AffineRecord((3,), (1,), 0, (2,), 2)),
    (AffineRecord((2, 2), (1, 1), 0, (4, 1), 1), AffineRecord((2,), (0,), 3, (-4,), 7)),
    (AffineRecord((3,), (-1,), 5, (-2,), 6), AffineRecord((2,), (1,), 0, (2,), 1)),
    (AffineRecord((0,), (1,), 6, (1,), 10),),
)


def update_functions(records):
    indices = []
    destinations = []
    for record in records:
        pairs = [(
            record.source_offset + sum(index * stride for index, stride in zip(coordinate, record.source_strides)),
            record.destination_offset + sum(index * stride for index, stride in zip(coordinate, record.destination_strides)),
        ) for coordinate in np.ndindex(record.logical_shape)]
        indices.append((jnp.asarray([source for source, _ in pairs], dtype=jnp.int32),
                        jnp.asarray([destination for _, destination in pairs], dtype=jnp.int32)))
        destinations.extend(destination for _, destination in pairs)
    assert len(destinations) == len(set(destinations))

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=records)

    def reference(source, base, alpha, beta):
        alpha = alpha[..., None] if alpha.ndim else alpha
        beta = beta[..., None] if beta.ndim else beta
        result = base
        for source_indices, destination_indices in indices:
            contribution = alpha * source[..., source_indices] + beta * base[..., destination_indices]
            result = result.at[..., destination_indices].set(contribution.astype(base.dtype))
        return result

    return execute, reference


def operands(dtype, batch_shape):
    source = (jnp.arange(prod(batch_shape) * 6) % 5).astype(dtype).reshape((*batch_shape, 6))
    base = (jnp.arange(prod(batch_shape) * 10) % 7).astype(dtype).reshape((*batch_shape, 10))
    if dtype == "complex64":
        source = source + 1j * (source - 2)
        base = base + 1j * (3 - base)
    return source, base, jnp.asarray(2, dtype=dtype), jnp.full(batch_shape, 3, dtype=dtype)


def check_joint_derivatives(execute, reference, arguments):
    directions = tuple(jnp.ones_like(value) for value in arguments)
    actual = jax.jit(lambda *values: jax.jvp(execute, values, directions))(*arguments)
    expected = jax.jvp(reference, arguments, directions)
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=2e-6, atol=1e-6)
    cotangent = arguments[1] + 1
    actual_gradient = jax.jit(jax.vjp(execute, *arguments)[1])(cotangent)
    expected_gradient = jax.vjp(reference, *arguments)[1](cotangent)
    for observed, wanted in zip(actual_gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(observed, wanted, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("records", LAYOUTS)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("batch_shape", [(), (2, 3), (0,)])
def test_multirecord_update_joint_jvp_vjp(records, dtype, batch_shape):
    check_joint_derivatives(*update_functions(records), operands(dtype, batch_shape))


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sequential_updates_with_overlapping_selections(dtype):
    first, reference_first = update_functions((AffineRecord((3,), (1,), 0, (1,), 1),))
    second, reference_second = update_functions((AffineRecord((3,), (-1,), 5, (1,), 2),))

    def execute(source, base, alpha, beta):
        return second(source, first(source, base, alpha, beta), alpha, beta)

    def reference(source, base, alpha, beta):
        return reference_second(source, reference_first(source, base, alpha, beta), alpha, beta)

    check_joint_derivatives(execute, reference, operands(dtype, (2, 3)))


@pytest.mark.parametrize("active,target", [(0, "accumulation"), (1, "update"), (2, "dot"), (3, "dot")])
def test_update_transpose_uses_existing_native_operations(active, target):
    arguments = operands("float32", ())
    execute, reference = update_functions(LAYOUTS[0])

    def bind(function, value):
        values = list(arguments)
        values[active] = value
        return function(*values)

    transpose = jax.jit(jax.linear_transpose(lambda value: bind(execute, value), arguments[active]))
    expected = jax.vjp(lambda value: bind(reference, value), arguments[active])[1](arguments[1])[0]
    np.testing.assert_allclose(transpose(arguments[1])[0], expected, rtol=2e-6, atol=1e-6)
    lowered = transpose.lower(arguments[1]).as_text()
    assert f"tensor0_stride_{target}_f32_cpu_v1" in lowered
    assert "stablehlo.gather" not in lowered
    assert "stablehlo.scatter" not in lowered
