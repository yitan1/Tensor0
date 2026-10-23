"""SVD algorithm selection and per-sector custom differentiation contract."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensor0 as t0

@pytest.fixture(autouse=True)
def enable_x64():
    with jax.enable_x64(True):
        yield


def tensor(empty=False):
    a = t0.space(t0.FermionParity, {} if empty else {0: 3, 1: 2})
    b = t0.space(t0.FermionParity, {} if empty else {0: 2, 1: 3})
    h = t0.hom((a,), (b,))
    return t0.from_blocks(h, {} if empty else {
        (0,): jnp.array([[3, 1j], [1, 2], [0, 1]], dtype=jnp.complex128),
        (1,): jnp.array([[4, 1j, 0], [0, 2, 1]], dtype=jnp.complex128),
    }, dtype=jnp.complex128)


@pytest.mark.parametrize('algorithm', [None, 'qr', 'divide_and_conquer'])
def test_tensor_interfaces(algorithm):
    x = tensor()
    u, s, vh = t0.svd_compact(x, algorithm=algorithm)
    for sector, block in x.blocks():
        np.testing.assert_allclose((u.block(sector) * s.diag().block(sector)) @ vh.block(sector), block, atol=1e-12)
    full = t0.svd_full(x, algorithm=algorithm)
    for sector, block in x.blocks():
        np.testing.assert_allclose(full[0].block(sector) @ full[1].block(sector) @ full[2].block(sector), block, atol=1e-12)
    np.testing.assert_allclose(t0.svd_vals(x, algorithm=algorithm).storage.data, s.storage.data, atol=1e-12)
    np.testing.assert_allclose(t0.svd_trunc(x, algorithm=algorithm)[1].storage.data, s.storage.data, atol=1e-12)


def test_empty_tensor():
    x = tensor(True)
    def unused_kernel(block):
        pytest.fail('empty tensor must not call the matrix kernel')
    for algorithm in (None, unused_kernel):
        u, s, vh = t0.svd_compact(x, algorithm=algorithm)
        assert u.storage.data.size == s.storage.data.size == vh.storage.data.size == 0
    assert t0.svd_vals(x).storage.data.size == 0
    assert all(t.storage.data.size == 0 for t in t0.svd_full(x))
    assert t0.svd_trunc(x)[1].storage.data.size == 0


def test_callable_jit_ad():
    x = tensor()
    kernel = partial(t0.matrix_svd, algorithm='qr')
    def values(data):
        y = t0.TensorMap(x.space, data)
        return t0.svd_compact(y, algorithm=kernel)[1].storage.data
    np.testing.assert_allclose(jax.jit(values)(x.storage.data), values(x.storage.data), atol=1e-12)
    tangent = jnp.ones_like(x.storage.data)
    assert jnp.all(jnp.isfinite(jax.jvp(values, (x.storage.data,), (tangent,))[1]))
    assert jnp.all(jnp.isfinite(jax.vjp(values, x.storage.data)[1](jnp.ones_like(values(x.storage.data)))[0]))
    t0.svd_trunc(x, algorithm=kernel, trunc=t0.truncrank(2))
    for f in (t0.svd_vals, t0.svd_full):
        with pytest.raises(ValueError):
            f(x, algorithm=kernel)


def test_errors():
    with pytest.raises(ValueError):
        t0.svd_compact(tensor(True), algorithm='invalid')
    if jax.default_backend() == 'cpu':
        with pytest.raises(NotImplementedError):
            t0.matrix_svd(jnp.eye(3), algorithm='jacobi')


def test_custom_rule_and_contract_checks():
    @jax.custom_jvp
    def kernel(block):
        return tuple(t0.matrix_svd(block))

    @kernel.defjvp
    def rule(primals, tangents):
        result = kernel(primals[0])
        return result, tuple(jnp.zeros_like(a) for a in result)

    x = tensor()
    def values(data):
        return t0.svd_compact(t0.TensorMap(x.space, data), algorithm=kernel)[1].storage.data
    np.testing.assert_array_equal(jax.jvp(values, (x.storage.data,), (jnp.ones_like(x.storage.data),))[1], 0)
    with pytest.raises(ValueError, match='shapes'):
        t0.svd_compact(x, algorithm=lambda a: (a, jnp.ones(1), a))
    with pytest.raises(TypeError, match='dtypes'):
        t0.svd_compact(x, algorithm=lambda a: tuple(v.astype(jnp.complex128) for v in t0.matrix_svd(a)))
