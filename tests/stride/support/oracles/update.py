"""Update layouts, disjoint address checks and differentiable reference."""

from functools import cache

import jax.numpy as jnp
import numpy as np

from tensor0._stride._jax import update_p
from tensor0._stride._layout import AffineRecord

from tests.stride.support.ad import check_coefficient_ad


def address_pairs(record):
    return tuple(((record.source_offset + sum((axis * stride for axis, stride in zip(coordinates, record.source_strides))), record.destination_offset + sum((axis * stride for axis, stride in zip(coordinates, record.destination_strides)))) for coordinates in np.ndindex(record.logical_shape)))


def assert_disjoint_writes(records):
    destinations = [destination for record in records for _, destination in address_pairs(record)]
    assert len(destinations) == len(set(destinations))


@cache
def update_functions(records):
    assert_disjoint_writes(records)
    indices = tuple(((np.asarray([source for source, _ in address_pairs(record)], dtype=np.int32), np.asarray([destination for _, destination in address_pairs(record)], dtype=np.int32)) for record in records))

    def execute(source, base, alpha, beta):
        return update_p.bind(source, base, alpha, beta, records=records)

    def reference(source, base, alpha, beta):
        factors = tuple((value.reshape(()) if value.shape == (1,) else value for value in (alpha, beta)))
        alpha, beta = tuple((value[..., None] if value.ndim else value for value in factors))
        result = base
        for source_indices, destination_indices in indices:
            contribution = alpha * source[..., source_indices] + beta * base[..., destination_indices]
            if not jnp.iscomplexobj(base):
                contribution = jnp.real(contribution)
            result = result.at[..., destination_indices].set(contribution.astype(base.dtype))
        return result
    return (execute, reference)


LAYOUTS = [
    (AffineRecord((3,), (1,), 0, (2,), 1), AffineRecord((3,), (1,), 0, (2,), 2)),
    (AffineRecord((2, 2), (1, 1), 0, (4, 1), 1), AffineRecord((2,), (0,), 3, (-4,), 7)),
    (AffineRecord((3,), (-1,), 5, (-2,), 6), AffineRecord((2,), (1,), 0, (2,), 1)),
    (AffineRecord((), (), 0, (), 2), AffineRecord((), (), 0, (), 5)),
    (AffineRecord((2,), (1,), 1, (1,), 2), AffineRecord((0,), (1,), 6, (1,), 10)),
    (AffineRecord((1, 1), (2 ** 63 - 1,) * 2, 0, (2 ** 63 - 1,) * 2, 1), AffineRecord((), (), 3, (), 4)),
    (AffineRecord((0,), (1,), 6, (1,), 10), AffineRecord((0,), (1,), 6, (1,), 10)),
    (),
]


def compare(run, reference, arguments):
    check_coefficient_ad(run, reference, arguments, coefficient_count=2, complex_cotangent=1 + 2j)
