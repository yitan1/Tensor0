"""Bounded CUDA owner/fiber benchmark of the current Tensor0 stride primitives.

Run against a CUDA-enabled Tensor0 installation, not a CPU fallback. These are
synchronized end-to-end hot calls, not isolated CUDA kernel timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import jax
import numpy as np

from tensor0 import _native
from tensor0._stride import _jax
from tensor0._stride._layout import AffineRecord


# Fixed workloads exercise the owner and fiber axes without creating a tuning API.
SUM_CASES = ((1, 64), (1, 4096), (128, 4096), (1024, 64))


def _measure(name, operation, host_args, expected, device, warmups, repeats):
    args = tuple(jax.device_put(value, device) for value in host_args)
    for value in args:
        value.block_until_ready()
    start = time.perf_counter_ns()
    executable = jax.jit(operation).lower(*args).compile()
    compile_ms = (time.perf_counter_ns() - start) / 1e6
    result = executable(*args)
    result.block_until_ready()
    if result.devices() != {device}:
        raise RuntimeError(f"{name}: result was not produced on the selected CUDA device")
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-6, atol=2e-6,
                               err_msg=name)
    for _ in range(warmups):
        executable(*args).block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        executable(*args).block_until_ready()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    return {"case": name, "compile_ms": compile_ms,
            "median_us": statistics.median(samples), "samples_us": samples}


def _workloads(device, warmups, repeats):
    results = []
    batch = 2
    for operation_name in ("accumulation", "reduction"):
        for owners, fiber in SUM_CASES:
            record = AffineRecord((owners, fiber), (fiber, 1), 0, (1, 0), 0)
            # Exact binary fractions and bounded values keep independent sums
            # well conditioned while exercising nonconstant input buffers.
            source = ((np.arange(batch * owners * fiber) % 7 - 3) / 4).astype(np.float32)
            source = source.reshape(batch, owners * fiber)
            expected = source.reshape(batch, owners, fiber).astype(np.float64).sum(axis=-1).astype(np.float32)
            if operation_name == "accumulation":
                call = lambda value, r=record, n=owners: _jax.accumulation_p.bind(
                    value, records=(r,), output_size=n, dtype=value.dtype,
                    coefficient_records=())
            else:
                call = lambda value, r=record, n=owners: _jax.reduction_p.bind(
                    value, records=(r,), output_shapes=((n, 1),),
                    reduction_axes=((False, True),), output_size=n,
                    dtype=value.dtype, coefficient_records=())
            results.append(_measure(f"{operation_name}/{owners}x{fiber}", call,
                                    (source,), expected, device, warmups, repeats))

    for operation_name in ("copy", "update"):
        count = 4096
        record = AffineRecord((count,), (1,), 0, (1,), 0)
        source = ((np.arange(batch * count) % 11 - 5) / 4).astype(np.float32).reshape(batch, count)
        if operation_name == "copy":
            call = lambda value: _jax.copy_p.bind(value, records=(record,),
                                                    output_size=count, dtype=value.dtype)
            args, expected = (source,), source.copy()
        else:
            base = np.full_like(source, 2)
            alpha, beta = np.asarray(0.5, np.float32), np.asarray(-0.25, np.float32)
            call = lambda value, old, a, b: _jax.update_p.bind(
                value, old, a, b, records=(record,))
            args, expected = (source, base, alpha, beta), 0.5 * source - 0.25 * base
        results.append(_measure(f"{operation_name}/{count}", call,
                                args, expected, device, warmups, repeats))

    for name, length, records, complex_input in (
        ("dot/65536", 65536, (AffineRecord((65536,), (1,), 0, (1,), 0),), False),
        ("dot/4records", 4096,
         tuple(AffineRecord((1024,), (1,), i * 1024, (1,), i * 1024) for i in range(4)), False),
        ("dotc/4096", 4096, (AffineRecord((4096,), (1,), 0, (1,), 0),), True),
    ):
        positions = np.arange(batch * length).reshape(batch, length)
        left = ((positions % 7 - 3) / 4).astype(np.float32)
        right = ((positions % 11 - 5) / 8).astype(np.float32)
        if complex_input:
            left = left + 1j * ((positions % 5 - 2) / 4)
            right = right + 1j * ((positions % 3 - 1) / 8)
            left, right = left.astype(np.complex64), right.astype(np.complex64)
        # Sum each record independently, then add in record order; no native
        # implementation is used to calculate the correctness reference.
        expected = np.zeros(batch, np.complex128 if complex_input else np.float64)
        for record in records:
            begin = record.source_offset
            end = begin + record.logical_shape[0]
            a = left[:, begin:end].astype(expected.dtype)
            b = right[:, begin:end].astype(expected.dtype)
            expected += (np.conj(a) * b if complex_input else a * b).sum(axis=1)
        expected = expected.astype(left.dtype)
        call = lambda a, b, r=records, c=complex_input: _jax.dot_p.bind(
            a, b, records=r, conjugate_left=c, dtype=a.dtype)
        results.append(_measure(name, call, (left, right), expected,
                                device, warmups, repeats))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be nonnegative and repeats must be positive")
    if not _native._stride_cuda_available():
        parser.error("Tensor0 was built without its CUDA stride backend")
    try:
        device = jax.devices("cuda")[0]
    except (RuntimeError, IndexError) as error:
        parser.error(f"a JAX CUDA device is required: {error}")
    native_path = Path(_native.__file__)
    native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
    results = _workloads(device, args.warmups, args.repeats)
    print(json.dumps({"device": str(device), "jax": jax.__version__,
                      "native": str(native_path), "native_sha256": native_hash,
                      "warmups": args.warmups,
                      "repeats": args.repeats, "results": results}, indent=2))


if __name__ == "__main__":
    main()
