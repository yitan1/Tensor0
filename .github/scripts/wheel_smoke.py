"""Verify installed public CPU operations; run with python -I outside the checkout."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys
from urllib.parse import unquote, urlparse
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
import tensor0
from tensor0 import TensorMap, U1Irrep, Vect, from_dense, hom, space, to_dense
import tensor0._native as native

assert Path(tensor0.__file__).resolve().is_relative_to(Path(sys.prefix))
assert Path(native.__file__).resolve().is_relative_to(Path(sys.prefix))
assert sys.flags.isolated and sys.prefix != sys.base_prefix
assert sys.implementation.name == 'cpython' and sys.version_info.releaselevel == 'final', sys.version
assert jax.__version__ == importlib.metadata.version('jaxlib') == '0.11.2'
assert native._stride_ffi_available()
assert native._stride_ffi_build_versions() == ('0.11.2', '0.11.2')
assert jax.default_backend() == 'cpu'
# Release jobs supply the original wheel; manual smoke callers need no argument.
if len(sys.argv) > 1:
    wheel_arg, = sys.argv[1:]
    wheel = Path(wheel_arg)
    version = wheel.name.split('-')[1]
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    assert wheel.name.split('-')[2:4] == [abi, abi]
    assert tensor0.__version__ == native.native_version() == importlib.metadata.version('tensor0') == version
    direct = json.loads(importlib.metadata.distribution('tensor0').read_text('direct_url.json'))
    assert not direct.get('dir_info')
    url = urlparse(direct['url'])
    assert url.scheme == 'file' and Path(unquote(url.path)).name == wheel.name
    assert direct['archive_info']['hashes']['sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        binary, = (name for name in archive.namelist()
                   if name.startswith('tensor0/_native') and name.endswith('.so'))
        assert Path(native.__file__).read_bytes() == archive.read(binary)
    print('Exact installed wheel:', wheel.name, direct)
target = hom((Vect(2),), (Vect(2),))
data = jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2)
tensor = from_dense(target, data)
np.testing.assert_allclose(to_dense(tensor), data)
np.testing.assert_allclose(to_dense(tensor @ tensor), data @ data)
np.testing.assert_allclose(to_dense(tensor.permute(((1,), (0,)))), data.T)
composed = jax.jit(lambda a: a @ a)(tensor)
np.testing.assert_allclose(to_dense(composed), data @ data)
# vmap must reconstruct a TensorMap result with leading batch axes, not only arrays.
batch = jnp.stack((data, 2 * data))
batched = jax.jit(jax.vmap(lambda a: from_dense(target, a)))(batch)
assert isinstance(batched, TensorMap) and batched.storage.data.shape == (2, 4)
np.testing.assert_allclose(jax.vmap(to_dense)(batched), batch)
batched_composed = jax.jit(jax.vmap(lambda a: a @ a))(batched)
assert isinstance(batched_composed, TensorMap)
np.testing.assert_allclose(jax.vmap(to_dense)(batched_composed), batch @ batch)

def loss(array):
    candidate = from_dense(target, array)
    transformed = candidate.permute(((1,), (0,)))
    return jnp.sum(to_dense(transformed) ** 2)

value, gradient = jax.jit(jax.value_and_grad(loss))(data)
np.testing.assert_allclose(value, jnp.sum(data ** 2))
np.testing.assert_allclose(gradient, 2 * data)
factor = space(U1Irrep, {0: 2, 1: 1})
symmetric_space = hom((factor,), (factor,))
symmetric_data = jnp.arange(1, 6, dtype=jnp.float32)
symmetric = TensorMap(symmetric_space, symmetric_data)
symmetric_composed = jax.jit(lambda a: a @ a)(symmetric)
for sector, block in symmetric.blocks():
    np.testing.assert_allclose(symmetric_composed.block(sector), block @ block)
symmetric_batch = jnp.stack((symmetric_data, 2 * symmetric_data))
batched_symmetric = jax.jit(jax.vmap(lambda a: TensorMap(symmetric_space, a)))(symmetric_batch)
assert isinstance(batched_symmetric, TensorMap) and batched_symmetric.storage.data.shape == (2, 5)
np.testing.assert_allclose(batched_symmetric.storage.data, symmetric_batch)
batched_symmetric_composed = jax.jit(jax.vmap(lambda a: a @ a))(batched_symmetric)
assert isinstance(batched_symmetric_composed, TensorMap)
np.testing.assert_allclose(batched_symmetric_composed.storage.data,
                           jnp.stack((symmetric_composed.storage.data, 4 * symmetric_composed.storage.data)))

def symmetric_loss(array):
    candidate = TensorMap(symmetric_space, array)
    return sum(jnp.sum(block ** 2) for _, block in candidate.blocks())

symmetric_value, symmetric_gradient = jax.jit(jax.value_and_grad(symmetric_loss))(symmetric_data)
np.testing.assert_allclose(symmetric_value, jnp.sum(symmetric_data ** 2))
np.testing.assert_allclose(symmetric_gradient, 2 * symmetric_data)
print('Installed wheel Trivial/U1 CPU ops, jit, direct TensorMap vmap and grad OK:', tensor0.__file__)
