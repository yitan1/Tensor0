"""Packed trace fixtures and independent expected values."""

from types import SimpleNamespace

import jax.numpy as jnp

from tensor0._stride._tensor_ops import _strided_tensortrace


def scalar_trace(source, factors, dtype):
    return _strided_tensortrace(
        source, destination_size=1,
        source_subblocks=tuple(SimpleNamespace(sizes=(), strides=(), offset=index)
                              for index in range(source.shape[-1])),
        destination_subblocks=(SimpleNamespace(sizes=(), strides=(), offset=0),),
        entries=((index, 0, factor) for index, factor in enumerate(factors)),
        result_dtype=dtype, permutation=(), num_open_out=0, num_open_in=0, trace_count=0)


def packed_trace(source, factor, second):
    return _strided_tensortrace(
        source, destination_size=5,
        source_subblocks=(
            SimpleNamespace(sizes=(2, 0, 0), strides=(1, 1, 1), offset=12),
            SimpleNamespace(sizes=(2, 2, 2), strides=(4, 2, 1), offset=0),
            SimpleNamespace(sizes=(2, 2, 2), strides=(0, -2, -1), offset=3),
        ),
        destination_subblocks=(SimpleNamespace(sizes=(2,), strides=(1,), offset=1),
                              SimpleNamespace(sizes=(2,), strides=(-1,), offset=3)),
        entries=((0, 0, None), (1, 0, None), (2, 1, factor), (1, 1, second)),
        result_dtype=source.dtype, permutation=(), num_open_out=1, num_open_in=0, trace_count=1,
    )


def reference_trace(source, factor, second):
    first = source[..., jnp.asarray([[0, 3], [4, 7]])]
    broadcast = source[..., jnp.asarray([[3, 0], [3, 0]])]
    output = jnp.zeros((*source.shape[:-1], 5), dtype=source.dtype)
    output = output.at[..., 1:3].add(first.sum(axis=-1))
    contribution = (broadcast * factor).sum(axis=-1) + (first * second).sum(axis=-1)
    if not jnp.issubdtype(source.dtype, jnp.complexfloating):
        contribution = jnp.real(contribution)
    return output.at[..., jnp.asarray([3, 2])].add(contribution.astype(source.dtype))
