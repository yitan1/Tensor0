from __future__ import annotations

import jax.numpy as jnp
from jax import Array, random
from jax.typing import DTypeLike

from .. import _native
from ..structure.layout import get_blockstructure
from ..structure.spaces import (
    _as_hom_space,
    _as_product_space_input,
    fuse,
    hom,
    is_isomorphic,
    is_monomorphic,
    storage_dim,
)
from ._orthogonal import positive_qr
from .tensor_map import TensorMap, from_blocks

_TensorSpace = _native.ElementarySpace | _native.ProductSpace


def identity(
    space: _TensorSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Construct the identity endomorphism on an elementary or product space."""
    product = _as_product_space(space, "space", "identity")
    return _identity_block_morphism(product, product, dtype=dtype)


def isomorphism(
    codomain: _TensorSpace,
    domain: _TensorSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Construct a deterministic isomorphism between two isomorphic spaces."""
    codomain_product = _as_product_space(codomain, "codomain", "isomorphism")
    domain_product = _as_product_space(domain, "domain", "isomorphism")
    _require_isomorphic(codomain_product, domain_product, "isomorphism")
    return _identity_block_morphism(codomain_product, domain_product, dtype=dtype)


def unitary(
    codomain: _TensorSpace,
    domain: _TensorSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Construct a deterministic unitary between two isomorphic spaces."""
    codomain_product = _as_product_space(codomain, "codomain", "unitary")
    domain_product = _as_product_space(domain, "domain", "unitary")
    _require_isomorphic(codomain_product, domain_product, "unitary")
    return _identity_block_morphism(codomain_product, domain_product, dtype=dtype)


def isometry(
    codomain: _TensorSpace,
    domain: _TensorSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Construct a deterministic isometric embedding of domain into codomain."""
    codomain_product = _as_product_space(codomain, "codomain", "isometry")
    domain_product = _as_product_space(domain, "domain", "isometry")
    if not is_monomorphic(fuse(domain_product), fuse(codomain_product)):
        raise ValueError(
            "isometry() requires the domain to be monomorphic into the codomain",
        )
    return _identity_block_morphism(codomain_product, domain_product, dtype=dtype)


def random_normal(
    key: Array,
    space: _native.HomSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Sample packed reduced storage from JAX's normal distribution."""
    target_space = _as_hom_space(space, "random_normal")
    data = random.normal(
        key,
        shape=(storage_dim(target_space),),
        dtype=dtype,
    )
    return TensorMap(target_space, data)


def random_isometry(
    key: Array,
    codomain: _TensorSpace,
    domain: _TensorSpace,
    *,
    dtype: DTypeLike | None = None,
) -> TensorMap:
    """Sample a blockwise random isometry from domain to codomain."""
    codomain_product = _as_product_space(codomain, "codomain", "random_isometry")
    domain_product = _as_product_space(domain, "domain", "random_isometry")
    if not is_monomorphic(fuse(domain_product), fuse(codomain_product)):
        raise ValueError(
            "random_isometry() requires the domain to be monomorphic into "
            "the codomain",
        )
    if dtype is not None:
        normalized_dtype = jnp.dtype(dtype)
        if not (
            jnp.issubdtype(normalized_dtype, jnp.floating)
            or jnp.issubdtype(normalized_dtype, jnp.complexfloating)
        ):
            raise ValueError("random_isometry() requires a float or complex dtype")

    target = hom(codomain_product, domain_product)
    sample = random_normal(key, target, dtype=dtype)
    q_blocks = (
        (coupled, positive_qr(block)[0]) for coupled, block in sample.blocks()
    )
    return from_blocks(target, q_blocks, dtype=sample.dtype)


def _as_product_space(
    value: object,
    argument_name: str,
    function_name: str,
) -> _native.ProductSpace:
    normalized = _as_product_space_input(value, argument_name, function_name)
    if isinstance(normalized, _native.ProductSpace):
        return normalized
    return hom(normalized, ()).codomain


def _require_isomorphic(
    codomain: _native.ProductSpace,
    domain: _native.ProductSpace,
    function_name: str,
) -> None:
    if not is_isomorphic(fuse(codomain), fuse(domain)):
        raise ValueError(
            f"{function_name}() requires isomorphic codomain and domain spaces",
        )


def _identity_block_morphism(
    codomain: _native.ProductSpace,
    domain: _native.ProductSpace,
    *,
    dtype: DTypeLike | None,
) -> TensorMap:
    target = hom(codomain, domain)
    blocks = (
        (coupled, jnp.eye(block.row_dim, block.col_dim, dtype=dtype))
        for coupled, block in get_blockstructure(target).items()
    )
    return from_blocks(target, blocks, dtype=dtype)
