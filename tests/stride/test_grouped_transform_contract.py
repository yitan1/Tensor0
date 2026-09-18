"""Grouped transform contract: independent numerical and boundary regressions."""

from math import prod
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tensor0._stride import _tensor_ops as _transform

from ._support import native_available


pytestmark = pytest.mark.skipif(not native_available(), reason="native CPU stride is unavailable")

def block(shape, strides, offset):
    return SimpleNamespace(sizes=shape, strides=strides, offset=offset)

def test_direct_conversion_follows_multiplication():
    with jax.enable_x64():
        layout = SimpleNamespace(subblockstructure=(block((1,), (1,), 0),), total_dim=1)
        groups = (SimpleNamespace(src_indices=(0,), dst_indices=(0,), transform=np.asarray([[1.5]])),)
        source = jnp.asarray([16777217], dtype=jnp.int32)
        actual = _transform._strided_grouped_transform(source, layout, layout, jnp.float64, (0,), groups)
        np.testing.assert_array_equal(actual, [25165825.5])
        actual = _transform._strided_grouped_transform(source, layout, layout, jnp.float32, (0,), groups)
        np.testing.assert_array_equal(actual, [25165824])
        source = source.astype(jnp.float64)
        actual = _transform._strided_grouped_transform(source, layout, layout, jnp.float32, (0,), groups)
        np.testing.assert_array_equal(actual, [25165826])
        source = jnp.ones(1, dtype=jnp.float16)
        run = lambda values: _transform._strided_grouped_transform(values, layout, layout, jnp.float32, (0,), groups)
        primal, tangent = jax.jvp(run, (source,), (source,))
        assert primal.dtype == tangent.dtype == jnp.float32
        np.testing.assert_array_equal(tangent, [1.5])
        gradient = jax.vjp(run, source)[1](jnp.ones(1, dtype=jnp.float32))[0]
        assert gradient.dtype == source.dtype
        np.testing.assert_array_equal(gradient, [1.5])

@pytest.mark.parametrize('factor', [0, 1])
def test_scalar_matrix_coefficient_short_circuit(factor):
    layout = SimpleNamespace(subblockstructure=(block((), (), 0),), total_dim=1)
    groups = (SimpleNamespace(src_indices=(0,), dst_indices=(0,), transform=np.asarray([[factor]])),)
    source = jnp.asarray([jnp.nan])
    result = _transform._strided_grouped_transform(source, layout, layout, source.dtype, (), groups)
    if factor == 0:
        assert result[0] == 0
    else:
        assert np.isnan(result[0])

def fixture_layouts(complex_matrix=False):
    sources = (block((2, 3), (3, -1), 2), block((2, 3), (0, 1), 3), block((2, 3), (3, 1), 0), block((0, 3), (3, 1), 6))
    destinations = tuple((block((3, 2), (6, 3), offset) for offset in range(3))) + (block((3, 0), (6, 3), 18),)
    matrix = np.asarray([[1, 2, -1], [-1, 0.5, 2]], dtype=np.complex64 if complex_matrix else np.float32)
    if complex_matrix:
        matrix[0, 1] += 1j
    groups = (SimpleNamespace(src_indices=(0, 1, 0), dst_indices=(2, 0), transform=matrix), SimpleNamespace(src_indices=(2,), dst_indices=(1,), transform=np.asarray([[-2]])), SimpleNamespace(src_indices=(3,), dst_indices=(3,), transform=np.asarray([[3]])))
    return (SimpleNamespace(subblockstructure=sources, total_dim=6), SimpleNamespace(subblockstructure=destinations, total_dim=18), groups)

def indices(view):
    return [view.offset + sum((index * stride for index, stride in zip(coordinate, view.strides))) for coordinate in np.ndindex(tuple(view.sizes))]

def oracle(source, source_layout, destination_layout, dtype, permutation, groups):
    result = jnp.zeros((*source.shape[:-1], destination_layout.total_dim), dtype=dtype)
    for group in groups:
        source_shape = tuple(source_layout.subblockstructure[group.src_indices[0]].sizes)
        rows = jnp.stack([source[..., jnp.asarray(indices(source_layout.subblockstructure[index]), dtype=jnp.int32)] for index in group.src_indices], axis=-2)
        matrix = jnp.asarray(group.transform, dtype=dtype)
        if matrix.shape == (1, 1):
            factor = np.asarray(group.transform).reshape(())
            mapped = jnp.zeros_like(rows) if factor == 0 else rows if factor == 1 else matrix.reshape(()) * rows
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
                mapped = jnp.real(mapped)
            mapped = mapped.astype(dtype)
        else:
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating):
                rows = jnp.real(rows)
            mapped = matrix @ rows.astype(dtype)
        batch_shape = source.shape[:-1]
        for row, destination_index in enumerate(group.dst_indices):
            shaped = mapped[..., row, :].reshape((*batch_shape, *source_shape))
            axes = (*range(len(batch_shape)), *(len(batch_shape) + axis for axis in permutation))
            values = jnp.transpose(shaped, axes).reshape((*batch_shape, prod(source_shape)))
            result = result.at[..., jnp.asarray(indices(destination_layout.subblockstructure[destination_index]), dtype=jnp.int32)].set(values)
    return result

@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.complex64])
@pytest.mark.parametrize('batch_shape', [(), (2,), (2, 0)])
def test_direct_only_repeated_reads_ad_and_no_pack(dtype, batch_shape):
    source_layout, destination_layout, groups = fixture_layouts()
    groups = tuple((SimpleNamespace(src_indices=(source_index,), dst_indices=(destination_index,), transform=np.asarray([[factor]])) for source_index, destination_index, factor in ((1, 0, 1), (1, 1, -2), (0, 2, 0)))) + groups[2:]
    source = jnp.arange(prod(batch_shape) * 6, dtype=jnp.float32).astype(dtype).reshape((*batch_shape, 6))
    arguments = (source_layout, destination_layout, source.dtype, (1, 0), groups)
    run = lambda values: _transform._strided_grouped_transform(values, *arguments)
    expected = lambda values: oracle(values, *arguments)
    lowered = jax.jit(run).lower(source)
    text = lowered.as_text()
    assert text.count('stablehlo.custom_call') == 1
    assert 'stride_accumulation_' in text
    for name in ('stride_copy_', 'stride_update_', 'stablehlo.dot_general', 'stablehlo.gather', 'stablehlo.scatter'):
        assert name not in text
    np.testing.assert_array_equal(lowered.compile()(source), expected(source))
    tangent = jnp.ones_like(source)
    np.testing.assert_array_equal(jax.jvp(run, (source,), (tangent,))[1], jax.jvp(expected, (source,), (tangent,))[1])
    cotangent = jnp.ones((*batch_shape, 18), dtype=dtype)
    np.testing.assert_array_equal(jax.vjp(run, source)[1](cotangent)[0], jax.vjp(expected, source)[1](cotangent)[0])

def test_multi_tree_pack_conversion_is_unchanged():
    with jax.enable_x64():
        source_layout = SimpleNamespace(subblockstructure=(block((1,), (1,), 0),), total_dim=1)
        destination_layout = SimpleNamespace(subblockstructure=(block((1,), (1,), 0), block((1,), (1,), 1)), total_dim=2)
        groups = (SimpleNamespace(src_indices=(0,), dst_indices=(0, 1), transform=np.asarray([[1.5], [2.5]])),)
        run = jax.jit(lambda source: _transform._strided_grouped_transform(source, source_layout, destination_layout, jnp.float32, (0,), groups))
        source = jnp.asarray([16777217], dtype=jnp.float64)
        np.testing.assert_array_equal(run(source), [25165824, 41943040])
        text = run.lower(source).as_text()
        assert text.count('stablehlo.custom_call') == 2
        assert 'stride_accumulation_' not in text
        assert 'stride_update_' not in text

@pytest.mark.parametrize('batch_shape', [(), (2,), (0,), (2, 0)])
def test_no_groups(batch_shape):
    layout = SimpleNamespace(subblockstructure=(), total_dim=0)
    source = jnp.zeros((*batch_shape, 0))
    run = lambda values: _transform._strided_grouped_transform(values, layout, layout, values.dtype, (), ())
    result = jax.jit(run)(source)
    assert result.shape == source.shape
    np.testing.assert_array_equal(jax.vjp(run, source)[1](result)[0], source)

def test_pack_unpack_records_and_shapes(monkeypatch):
    source_layout, destination_layout, groups = fixture_layouts()
    calls = []
    direct_calls = []
    update_calls = []

    def capture(source, **parameters):
        calls.append((source.dtype, parameters))
        return jnp.zeros((*source.shape[:-1], parameters['output_size']), dtype=parameters['dtype'])
    monkeypatch.setattr(_transform.copy_p, 'bind', capture)

    def capture_direct(source, *coefficients, **parameters):
        direct_calls.append((source.dtype, coefficients, parameters))
        return jnp.zeros((*source.shape[:-1], parameters['output_size']), dtype=parameters['dtype'])

    def capture_update(source, base, alpha, beta, **parameters):
        update_calls.append((source, base, alpha, beta, parameters))
        return base
    monkeypatch.setattr(_transform.accumulation_p, 'bind', capture_direct)
    monkeypatch.setattr(_transform.update_p, 'bind', capture_update)
    _transform._strided_grouped_transform(jnp.arange(6, dtype=jnp.float16), source_layout, destination_layout, jnp.float32, (1, 0), groups)
    assert len(calls) == len(direct_calls) == len(update_calls) == 1
    assert calls[0][0] == direct_calls[0][0] == jnp.float16
    assert calls[0][1]['dtype'] == jnp.float32
    assert tuple((record.destination_offset for record in calls[0][1]['records'])) == (0, 6, 12)
    assert tuple((record.source_offset for record in update_calls[0][-1]['records'])) == (0, 6)
    assert update_calls[0][0].shape == (12,)
    assert update_calls[0][1].shape == (18,)
    records = (*direct_calls[0][-1]['records'], *update_calls[0][-1]['records'])
    destinations = [record.destination_offset + sum((index * stride for index, stride in zip(coordinate, record.destination_strides))) for record in records for coordinate in np.ndindex(record.logical_shape)]
    assert sorted(destinations) == list(range(18))
    assert all((not hasattr(record, 'scale') for record in records))

@pytest.mark.parametrize('side', ['source', 'destination'])
def test_group_shape_errors(side):
    source_layout, destination_layout, groups = fixture_layouts()
    layout = source_layout if side == 'source' else destination_layout
    subblocks = list(layout.subblockstructure)
    index = 1 if side == 'source' else 0
    original = subblocks[index]
    subblocks[index] = block((1, 6), original.strides, original.offset)
    layout.subblockstructure = tuple(subblocks)
    with pytest.raises(ValueError, match=f'generic tree transform {side} subblock shapes are inconsistent'):
        _transform._strided_grouped_transform(jnp.arange(6, dtype=jnp.float32), source_layout, destination_layout, jnp.float32, (1, 0), groups)

def test_native_pack_address_validation():
    source_layout, destination_layout, groups = fixture_layouts()
    with pytest.raises(Exception, match='(?i)(bounds|storage)'):
        _transform._strided_grouped_transform(jnp.ones(5), source_layout, destination_layout, jnp.float32, (1, 0), groups).block_until_ready()
