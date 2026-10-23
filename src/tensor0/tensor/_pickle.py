"""Internal reconstruction for trusted TensorMap pickles."""
import jax
import jax.numpy as jnp
import numpy as np

from .. import _native
from ..structure.spaces import hom, space


_ATOMS = {
    atom.static_key: atom
    for atom in (_native.Trivial, _native.U1Irrep, _native.SU2Irrep,
                 _native.Z2Irrep, _native.Z3Irrep, _native.Z4Irrep,
                 _native.FermionParity)
}


def _restore_spec(key):
    if key in _ATOMS:
        return _ATOMS[key]
    factors = iter(key[1])
    result = _restore_spec(next(factors))
    for factor in factors:
        result = result @ _restore_spec(factor)
    return result


def _factors(product):
    # The constructor dualizes labels, so save the pre-dual labels.
    return tuple((factor.is_dual,
                  (factor.dual() if factor.is_dual else factor).sectors)
                 for factor in product)


def reduce_tensor(tensor):
    from .tensor_map import TensorMap

    if type(tensor) is not TensorMap:
        raise TypeError("pickle supports ordinary TensorMap values only")
    if tensor.batch_shape:
        raise ValueError("pickling batched TensorMaps is not supported")
    return restore_tensor, (
        tensor.space.sector_spec.static_key,
        _factors(tensor.codomain), _factors(tensor.domain),
        np.array(tensor.storage.data, copy=True),
    )


def restore_tensor(key, codomain, domain, data):
    from .tensor_map import TensorMap

    if np.dtype(jax.dtypes.canonicalize_dtype(data.dtype)) != data.dtype:
        raise ValueError("JAX configuration cannot preserve dtype; enable x64 before loading")
    spec = _restore_spec(key)
    def restore_factors(factors):
        return [space(spec, dict(labels), dual=dual) for dual, labels in factors]
    target = hom(restore_factors(codomain), restore_factors(domain), sector_type=spec)
    return TensorMap(target, jnp.array(data, copy=True))
