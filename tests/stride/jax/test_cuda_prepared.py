"""Compare equivalent raw layouts after shared locality sorting and fusion.

Singleton axes change initial locality ranking before removal; the two prepared
layouts need not have the same iteration or summation tree. Independent CUDA
executions must still implement the same logical operation.
"""

from itertools import product

import jax
import numpy as np
import pytest

from tensor0 import _native
from tests.stride.support.availability import cuda_device_or_skip
from tensor0._stride._ffi import _calls
from tensor0._stride._ffi._descriptor import encode_layout, encode_reduction_layout
from tensor0._stride._layout import AffineRecord


DTYPES = ("float32", "float64", "complex64", "complex128")
EMPTY = AffineRecord((2, 0, 3), (3, 3, 1), 0, (3, 3, 1), 0)


@pytest.fixture
def cuda_device():
    return cuda_device_or_skip()


def _interleave(values, barrier):
    return tuple(item for i, value in enumerate(values)
                 for item in ((barrier, value) if i else (value,)))


def _barrier(record):
    return AffineRecord(_interleave(record.logical_shape, 1),
                        _interleave(record.source_strides, -7919), record.source_offset,
                        _interleave(record.destination_strides, 3571), record.destination_offset)


def _ranks(words, reduction):
    cursor, ranks = 4, []
    for _ in range(words[3]):
        rank = words[cursor]
        ranks.append(rank)
        cursor += 3 + (5 if reduction else 3) * rank
    assert cursor == len(words)
    return tuple(ranks)


def _layouts(operation, records, source_size, output_size, axes=None):
    reduction = operation == "reduction"

    def encode(recs, roles):
        if not reduction:
            return encode_layout(recs, source_size=source_size, output_size=output_size)
        shapes = tuple(tuple(1 if flag else extent for extent, flag in
                             zip(record.logical_shape, flags, strict=True))
                       for record, flags in zip(recs, roles, strict=True))
        return encode_reduction_layout(recs, output_shapes=shapes, reduction_axes=roles,
                                       source_size=source_size, output_size=output_size)

    barriers = tuple(_barrier(record) for record in records)
    layouts = (encode(records, axes),
               encode(barriers, None if axes is None else
                      tuple(_interleave(flags, False) for flags in axes)))
    prepared = []
    for layout in layouts:
        words = layout.view("<i8") if reduction else layout
        prepared.append(_native._stride_prepare_layout(operation, tuple(map(int, words))))
    for prepared_words in prepared:
        assert _native._stride_prepare_layout(operation, prepared_words) == prepared_words
    return layouts


def _pattern(size, dtype, batches=2):
    # Copy/Update retain extreme signed bit patterns for exact mapping checks.
    large = 2.0 ** (24 if dtype in ("float32", "complex64") else 53)
    cycle = np.array([large, 1, -large, 3, -0.0, .25, -2, -.125], dtype=dtype)
    values = np.resize(cycle, batches * size).reshape(batches, size).copy()
    if dtype.startswith("complex"):
        values.imag = np.resize(cycle.real[::-1], values.size).reshape(values.shape)
    return values


def _bounded_pattern(size, dtype, batches=2):
    values = np.resize(np.array([11, 1, -9, 3, 0, .25, -2, -.125], dtype),
                       (batches, size)).copy()
    if dtype.startswith("complex"):
        values.imag = np.resize(np.array([1, -2, 5, -3, .25, 4], float), values.shape)
    return values


def _compare(device, layouts, operation, *host_args, exact=False):
    args = tuple(jax.device_put(value, device) for value in host_args)
    # Independent closures with explicit descriptors: no monkeypatch or cache
    # mutation can accidentally make both calls use the same prepared layout.
    def compile_layout(layout):
        return jax.jit(lambda *values: operation(layout, *values))
    fused = compile_layout(layouts[0])(*args)
    baseline = compile_layout(layouts[1])(*args)
    assert fused.devices() == baseline.devices() == {device}
    actual, expected = np.asarray(fused), np.asarray(baseline)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    if exact:
        np.testing.assert_array_equal(actual.reshape(-1).view(np.uint8),
                                      expected.reshape(-1).view(np.uint8))
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    return actual


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sign", [-1, 1])
def test_copy_signed_storage_bits(cuda_device, dtype, sign):
    records = (EMPTY, AffineRecord((2, 3), (sign * 3, sign), 5 if sign < 0 else 0,
                                   (-6, -2), 12))
    layouts = _layouts("copy", records, 8, 16)
    with jax.enable_x64():
        _compare(cuda_device, layouts,
                 lambda layout, s: _calls.execute_copy(s, layout=layout, output_size=16,
                                                       platform="cuda"), _pattern(8, dtype), exact=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_update_all_coefficient_branches_and_base(cuda_device, dtype):
    records = (EMPTY, AffineRecord((2, 3), (-3, -1), 5, (6, 2), 1),
               AffineRecord((2, 2), (2, 1), 6, (-4, -2), 18))
    layouts = _layouts("update", records, 10, 21)
    general = 1.25 + .5j if dtype.startswith("complex") else 1.25
    pairs = list(product((0, 1, general), repeat=2))
    alpha, beta = (np.asarray([pair[i] for pair in pairs], dtype) for i in (0, 1))
    with jax.enable_x64():
        _compare(cuda_device, layouts,
                 lambda layout, s, b, a, c: _calls.execute_update(
                     s, b, a, c, layout=layout, platform="cuda"),
                 _pattern(10, dtype, 9), _pattern(21, dtype, 9), alpha, beta, exact=True)


ACCUMULATION = [
    pytest.param((2, 3, 2), (6, 2, 1), 12, id="injective"),
    pytest.param((2, 3, 2), (0, 0, 1), 2, id="zero-stride-fiber"),

]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,destination,size", ACCUMULATION)
def test_accumulation_paths_and_sparse_coefficients(cuda_device, dtype, shape, destination, size):
    record = AffineRecord(shape, (shape[1] * shape[2], shape[2], 1), 0, destination, 0)
    records = (EMPTY, record, record)
    layouts = _layouts("accumulation", records, 12, size + 2)
    source = _bounded_pattern(12, dtype, 3)
    coefficients = np.asarray([0, 1, 1.25 + .5j if dtype.startswith("complex") else 1.25], dtype)
    with jax.enable_x64():
        actual = _compare(cuda_device, layouts,
                          lambda layout, s, empty, factor: _calls.execute_accumulation(
                              s, (empty, factor), coefficient_records=(0, 2), layout=layout,
                              output_size=size + 2, platform="cuda"),
                          source, np.asarray(-7, dtype), coefficients)
    # Independent original-axis address enumeration: two overlapping records,
    # with a coefficient only on the latter and an empty coefficient record.
    reference = np.zeros((3, size + 2), dtype=dtype)
    for batch in range(3):
        for factor in (1, coefficients[batch]):
            for coordinates in product(*(range(extent) for extent in shape)):
                source_index = sum(index * stride for index, stride in
                                   zip(coordinates, record.source_strides, strict=True))
                output_index = sum(index * stride for index, stride in
                                   zip(coordinates, destination, strict=True))
                reference[batch, output_index] += factor * source[batch, source_index]
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sign,conjugate", list(product((-1, 1), (False, True))))
def test_dot_long_fiber_signed_and_conjugated(cuda_device, dtype, sign, conjugate):
    records = (AffineRecord((2, 1025), (sign * 1025, sign), 2049 if sign < 0 else 0,
                            (-1025, -1), 2049), EMPTY,
               AffineRecord((2, 3), (3, 1), 0, (3, 1), 7))
    layouts = _layouts("dot", records, 2050, 2050)
    right = np.resize(np.array([1, -.5, 2, .25, -1, 3, .125], dtype), (2, 2050))
    if dtype.startswith("complex"):
        right.imag = np.roll(right.real, 3, axis=-1)
    with jax.enable_x64():
        _compare(cuda_device, layouts,
                 lambda layout, a, b: _calls.execute_dot(
                     a, b, layout=layout, conjugate_left=conjugate, platform="cuda"),
                 _bounded_pattern(2050, dtype), right)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_reduction_roles_ordered_overlap_and_sparse_coefficients(cuda_device, dtype, reverse):
    sign = -1 if reverse else 1
    record = AffineRecord((2, 2, 2, 3), tuple(sign * s for s in (12, 6, 3, 1)),
                          23 if reverse else 0, (2, 1, 0, 0), 1)
    # Initialize the same destinations before the two fibers. The latter must
    # start with existing output, not reduce at zero and add a partial sum.
    initial = AffineRecord((2, 2), (2, 1), 24, (2, 1), 1)
    records = (initial, EMPTY, record, record)
    axes = ((False, False), (False, False, False),
            (False, False, True, True), (False, False, True, True))
    layouts = _layouts("reduction", records, 28, 7, axes)
    with jax.enable_x64():
        _compare(cuda_device, layouts,
                 lambda layout, s, empty, factor: _calls.execute_reduction(
                     s, (empty, factor), coefficient_records=(1, 3), layout=layout,
                     output_size=7, platform="cuda"),
                 _bounded_pattern(28, dtype, 3), np.asarray(-7, dtype),
                 np.asarray([0, 1, 1.25 + .5j if dtype.startswith("complex") else 1.25], dtype))

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", ("accumulation", "dot", "reduction"))
def test_column_major_cancellation_against_independent_sum(cuda_device, dtype, operation):
    import math

    dot = operation == "dot"
    # Raw row-major logical tuples visit source addresses 0, 2, 1, 3.
    # Sorting the column-major source creates one rank-one contiguous axis.
    record = AffineRecord((2, 2), (1, 2), 0, (1, 2) if dot else (0, 0), 0)
    layouts = _layouts(operation, (record,), 4, 4 if dot else 1,
                       ((True, True),) if operation == "reduction" else None)
    raw = layouts[0].view("<i8") if operation == "reduction" else layouts[0]
    canonical = _native._stride_prepare_layout(operation, tuple(map(int, raw)))
    assert _ranks(canonical, operation == "reduction") == (1,)
    assert tuple(map(int, raw)) != canonical
    large = 2**20
    source = np.asarray([[large, 1, -large, 3]], dtype)
    if dtype.startswith("complex"):
        source.imag = np.asarray([[7, -3, large, -large]], float)
    # Independent mathematical reference, not a second CUDA descriptor.
    order = (0, 2, 1, 3)
    expected = complex(math.fsum(float(source[0, i].real) for i in order),
                       math.fsum(float(source[0, i].imag) for i in order)
                       if dtype.startswith("complex") else 0)
    with jax.enable_x64():
        source_gpu = jax.device_put(source, cuda_device)
        one_gpu = jax.device_put(np.ones((1, 4), dtype), cuda_device)
        for layout in layouts:
            if dot:
                invoke = lambda: _calls.execute_dot(
                    source_gpu, one_gpu, layout=layout, conjugate_left=False, platform="cuda")
            elif operation == "reduction":
                invoke = lambda: _calls.execute_reduction(
                    source_gpu, layout=layout, output_size=1, platform="cuda")
            else:
                invoke = lambda: _calls.execute_accumulation(
                    source_gpu, layout=layout, output_size=1, platform="cuda")
            actual = np.asarray(jax.jit(invoke)())
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonfusable_dot_permutation_against_address_oracle(cuda_device, dtype):
    import math

    record = AffineRecord((3, 2), (1, 3), 0, (2, 1), 0)
    layouts = _layouts("dot", (record,), 6, 6)
    raw = tuple(map(int, layouts[0]))
    prepared = _native._stride_prepare_layout("dot", raw)
    assert _ranks(prepared, False) == (2,)
    assert prepared != raw
    values = np.asarray([[2**20, 1, -2**20, 3, -5, 7]], dtype)
    weights = np.asarray([[1, -1, 2, -2, .5, -.5]], dtype)
    if dtype.startswith("complex"):
        values = values + (1j * np.asarray([[7, -3, 2**20, -2**20, 1, 4]], dtype))
        weights = weights + 1j * np.asarray([[0, .5, 0, -.5, 1, 0]], dtype)
    products = [complex(values[0, i + 3 * j]) * complex(weights[0, 2 * i + j])
                for i, j in product(range(3), range(2))]
    reference = complex(math.fsum(value.real for value in products),
                        math.fsum(value.imag for value in products))
    with jax.enable_x64():
        left, right = (jax.device_put(x, cuda_device) for x in (values, weights))
        for layout in layouts:
            actual = np.asarray(jax.jit(lambda: _calls.execute_dot(
                left, right, layout=layout, conjugate_left=False, platform="cuda"))())
            np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-6)

@pytest.mark.parametrize("destination", [(1, 1), (3, 2)])
@pytest.mark.parametrize("coefficient", [0, 1])
def test_accumulation_rejects_unproven_map_before_gpu_launch(cuda_device, destination, coefficient):
    record = AffineRecord((2, 3), (3, 1), 0, destination, 0)
    layout = encode_layout((record,), source_size=6, output_size=8)
    with jax.enable_x64():
        data = jax.device_put(np.ones((0, 6), np.float32), cuda_device)
        with pytest.raises(ValueError, match="cannot prove injective output owners"):
            _calls.execute_accumulation(data, (jax.device_put(np.float32(coefficient), cuda_device),),
                                        coefficient_records=(0,), layout=layout,
                                        output_size=8, platform="cuda").block_until_ready()
