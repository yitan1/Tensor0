"""Focused affine-update regression benchmark; run as a Python module."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from tensor0._stride import StridedView, scale
from tensor0._stride._jax import accumulation_p, copy_p, update_p
from tensor0._stride._layout import AffineRecord


def measure(function, arguments, repeats):
    executable = jax.jit(function).lower(*arguments).compile()
    for _ in range(5):
        executable(*arguments).block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        executable(*arguments).block_until_ready()
        samples.append((time.perf_counter_ns() - start) / 1_000)
    memory = executable.memory_analysis()
    if memory is None:
        raise RuntimeError("the benchmark requires compiled memory statistics")
    return {
        "median_us": statistics.median(samples),
        "temporary_bytes": memory.temp_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
    }


def benchmark(size, repeats):
    results = {}
    data = jnp.arange(size * size, dtype=jnp.float32) / (size * size)
    for layout, strides in (("contiguous", (size, 1)), ("transpose", (1, size))):
        records = (AffineRecord((size, size), strides, 0, (size, 1), 0),)
        results[f"{layout}/convert_f16_f32"] = measure(
            lambda values: copy_p.bind(values, records=records, output_size=data.size,
                                       dtype=jnp.dtype(jnp.float32)),
            (data.astype(jnp.float16),), repeats,
        )
        for name, first, second in (("assign", 1, 0), ("accumulate", 1, 1),
                                    ("weighted", .75, -.25)):
            results[f"{layout}/{name}"] = measure(
                lambda old, source: update_p.bind(
                    source, old, jnp.asarray(first), jnp.asarray(second), records=records,
                ), (data, data), repeats,
            )
        operation = lambda values, factor: scale(
            StridedView(values, (size, size), strides, 0), factor,
        ).data
        for dtype in (np.float32, np.float64):
            results[f"{layout}/scale_{np.dtype(dtype).name}"] = measure(
                operation, (data, dtype(.75)), repeats,
            )
        results[f"{layout}/jvp_f64"] = measure(
            lambda values, factor, direction: jax.jvp(
                lambda source: operation(source, factor), (values,), (direction,),
            )[1], (data, np.float64(1e-46), jnp.full_like(data, 1e30)), repeats,
        )
    return results


def benchmark_leaves(size, repeats):
    results = {}
    tile = max(8, size // 8 * 8)
    layouts = (
        ("contiguous", (size, size), (size, 1)),
        ("transpose", (size, size), (1, size)),
        ("broadcast", (size, size), (1, 0)),
        ("rank4", (2, 3, tile, tile), (tile * tile, 2 * tile * tile, 1, tile)),
    )
    for layout, shape, strides in layouts:
        selected_size = int(np.prod(shape))
        source_size = 1 + sum((length - 1) * stride
                              for length, stride in zip(shape, strides))
        destination_strides = tuple(int(np.prod(shape[axis + 1:]))
                                    for axis in range(len(shape)))
        for dtype in ("float16", "float32", "complex64"):
            source = jnp.linspace(-1, 1, source_size, dtype=jnp.float32).astype(dtype)
            for partial in (False, True):
                offset = 8 if partial else 0
                base = jnp.ones((selected_size + 2 * offset,), dtype=dtype)
                for factor in (None, np.dtype(dtype).type(.75)):
                    records = (AffineRecord(shape, strides, 0, destination_strides, offset),)
                    name = f"{layout}/{dtype}/{partial}/{factor}"
                    results[f"{name}/fresh"] = measure(
                        lambda values: (
                            copy_p.bind(values, records=records, output_size=base.size,
                                        dtype=values.dtype) if factor is None else
                            accumulation_p.bind(values, jnp.asarray(factor), records=records,
                                                coefficient_records=(0,), output_size=base.size,
                                                dtype=values.dtype)
                        ), (source,), repeats,
                    )
                    for operation, first, second in (("overwrite", 1, 0),
                                                     ("scale", .75, 0),
                                                     ("weighted", .75, -.25)):
                        results[f"{name}/{operation}"] = measure(
                            lambda old, values: update_p.bind(
                                values, old, jnp.asarray(first if factor is None else first * factor),
                                jnp.asarray(second), records=records,
                            ), (base, source), repeats,
                        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=30)
    arguments = parser.parse_args()
    if arguments.size < 1 or arguments.repeats < 1:
        parser.error("size and repeats must be positive")
    with jax.enable_x64():
        results = benchmark(arguments.size, arguments.repeats)
        results.update(benchmark_leaves(arguments.size, arguments.repeats))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
