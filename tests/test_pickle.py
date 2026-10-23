"""Standard pickle support for trusted TensorMap workflows."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensor0 as t


def assert_tensor_equal(actual, expected):
    assert type(actual) is t.TensorMap
    assert actual.space == expected.space
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.storage.data, expected.storage.data)


@pytest.mark.parametrize('spec,labels', [
    (t.Trivial, [()]), (t.U1Irrep, [(-1,), (0,), (2,)]),
    (t.Z3Irrep, [(2,), (0,)]),
    (t.Z2Irrep, [(0,), (1,)]), (t.Z4Irrep, [(1,), (3,)]),
    (t.FermionParityU1Irrep, [(0, 0), (1, -1)]),
    (t.FermionParitySU2Irrep, [(0, 0), (1, 1)]), (t.SU2Irrep, [(0,), (1,), (2,)]),
    (t.FermionParity, [(0,), (1,)]),
    (t.FermionNumber, [(0, 0), (1, 1)]),
    (t.FermionParityU1SU2Irrep, [(0, 0, 0), (1, 1, 1)]),
    (t.U1Irrep @ t.SU2Irrep, [(0, 0), (2, 1)]),
])
def test_symmetry_roundtrip(spec, labels):
    v = t.space(spec, {k: i + 1 for i, k in enumerate(labels)})
    w = t.space(spec, {labels[0]: 3}, dual=True)
    h = t.hom([v, w], [w, v])
    values = jnp.arange(1, t.storage_dim(h) + 1, dtype='float32')
    data = values + 2j * values
    x = t.TensorMap(h, data)
    y = pickle.loads(pickle.dumps(x))
    assert_tensor_equal(y, x)


@pytest.mark.parametrize('protocol', sorted({pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL}))
def test_multiple_file(tmp_path, protocol):
    a = t.TensorMap(t.hom([t.Vect(2)], [t.Vect(2)]),
                    jnp.array([1., -2., 3., 4.], dtype='float32'))
    b = t.TensorMap(t.hom([], [], sector_type=t.Z3Irrep), jnp.array([3], dtype='int16'))
    v = t.space(t.U1Irrep, {1: 2}, dual=True)
    c = t.TensorMap(t.hom([v], [v]),
                    jnp.array([1j, 2 - 3j, -4, 5j], dtype='complex64'))
    path = tmp_path / 'tensors.pkl'
    with path.open('wb') as stream:
        pickle.dump({
            'psi': a, 'nested': [b, (c, a)],
            'model_params': {'L': 8, 'coupling': 1.5, 'boundary': 'periodic'},
            'dmrg_params': {'max_sweeps': 20, 'tolerance': 1.e-8},
        }, stream, protocol=protocol)
    with path.open('rb') as stream:
        result = pickle.load(stream)
    assert result['model_params'] == {'L': 8, 'coupling': 1.5, 'boundary': 'periodic'}
    assert result['dmrg_params'] == {'max_sweeps': 20, 'tolerance': 1.e-8}
    assert result['psi'] is result['nested'][1][1]
    for x, y in zip([a, b, c], [result['psi'], result['nested'][0], result['nested'][1][0]]):
        assert_tensor_equal(y, x)


@pytest.mark.parametrize('dtype', ['float64', 'complex128', 'int64', 'uint64'])
@pytest.mark.parametrize('empty', [False, True])
def test_precision(dtype, empty):
    with jax.enable_x64(True):
        h = t.hom([t.Vect(0 if empty else 2)], [])
        values = {
            'float64': [1 + 2**-40, -3 + 2**-40],
            'complex128': [1 + (1 + 2**-40) * 1j, -3j],
            'int64': [2**40 + 1, -(2**40 + 3)],
            'uint64': [2**40 + 1, 2**40 + 3],
        }[dtype]
        x = t.TensorMap(h, jnp.array([] if empty else values, dtype=dtype))
        payload = pickle.dumps(x)
        assert_tensor_equal(pickle.loads(payload), x)
        assert jax.config.read('jax_enable_x64')
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match='preserve dtype'):
            pickle.loads(payload)
        assert not jax.config.read('jax_enable_x64')


@pytest.mark.parametrize('spec', [t.U1Irrep, t.Z3Irrep, t.FermionNumber])
def test_empty_products(spec):
    for h in [t.hom([], [], sector_type=spec), t.hom([t.space(spec, {})], [])]:
        x = t.zeros(h)
        y = pickle.loads(pickle.dumps(x))
        assert_tensor_equal(y, x)


def test_batched_rejected():
    x = t.zeros(t.hom([t.Vect(2)], []))
    batched = jax.vmap(lambda _: x)(jnp.arange(2))
    with pytest.raises(ValueError, match='batched'):
        pickle.dumps(batched)


@pytest.mark.parametrize('dtype', ['bool', 'int8', 'int16', 'int32', 'uint8',
                                    'uint16', 'uint32', 'float16', 'float32',
                                    'complex64'])
def test_dtypes(dtype):
    x = t.TensorMap(t.hom([t.Vect(3)], []), jnp.arange(3).astype(dtype))
    y = pickle.loads(pickle.dumps(x))
    assert_tensor_equal(y, x)
