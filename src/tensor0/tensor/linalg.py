from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, SupportsFloat, overload

from jax import Array
import jax.numpy as jnp

from .. import _native
from ..structure.layout import get_blockstructure, get_sectorstructure
from ..structure.sector_dict import SectorDict
from ..structure.spaces import (
    _as_product_space_input,
    fuse,
    hom,
    is_isomorphic,
    is_monomorphic,
)
from ._blocks import pack_blocks
from ._tolerances import default_pseudoinverse_rtol, nonnegative_tolerance
from .diagonal import DiagonalTensorMap
from .tensor_map import TensorMap


def diagm(
    codomain: _native.ElementarySpace | _native.ProductSpace,
    domain: _native.ElementarySpace | _native.ProductSpace,
    values: Iterable[tuple[object, object]] | Mapping[Any, object],
) -> TensorMap:
    result_space = hom(
        _as_product_space_input(codomain, "codomain", "diagm"),
        _as_product_space_input(domain, "domain", "diagm"),
    )
    try:
        raw_value_blocks = SectorDict(values)
    except TypeError as error:
        raise TypeError("diagm() requires values to be a mapping or iterable") from error
    value_items = tuple(
        (coupled, jnp.asarray(value)) for coupled, value in raw_value_blocks.items()
    )
    dtype = (
        jnp.result_type(*(value for _coupled, value in value_items))
        if value_items
        else None
    )
    value_blocks = SectorDict(
        (coupled, jnp.asarray(value, dtype=dtype)) for coupled, value in value_items
    )
    blockstructures = get_blockstructure(result_space)
    block_arrays: list[tuple[tuple[int, ...], Array]] = []
    for coupled, block in blockstructures.items():
        sector_values = value_blocks.get(coupled)
        if sector_values is None:
            raise ValueError(f"missing data for diagonal block sector {coupled}")
        expected_shape = (min(block.row_dim, block.col_dim),)
        actual_shape = tuple(sector_values.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"diagonal block sector {coupled} has shape {actual_shape}; "
                f"expected {expected_shape}",
            )
        block_array = jnp.zeros((block.row_dim, block.col_dim), dtype=dtype)
        row_indices = jnp.arange(sector_values.shape[0])
        block_array = block_array.at[row_indices, row_indices].set(sector_values)
        block_arrays.append((coupled, block_array))

    for coupled, sector_values in value_blocks.items():
        if coupled in blockstructures:
            continue
        if sector_values.size != 0:
            raise ValueError(f"unexpected diagonal block sector {coupled}")

    return TensorMap(
        result_space,
        pack_blocks(result_space, SectorDict(block_arrays), dtype=dtype),
    )


@overload
def zero_like(tensor: TensorMap) -> TensorMap: ...


@overload
def zero_like(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def zero_like(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "zero_like")
    return tensor.zero_like()


@overload
def scale(tensor: TensorMap, alpha: object) -> TensorMap: ...


@overload
def scale(tensor: DiagonalTensorMap, alpha: object) -> DiagonalTensorMap: ...


def scale(
    tensor: TensorMap | DiagonalTensorMap,
    alpha: object,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "scale")
    return tensor.scale(alpha)


@overload
def add(
    t1: TensorMap,
    t2: TensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> TensorMap: ...


@overload
def add(
    t1: DiagonalTensorMap,
    t2: DiagonalTensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> DiagonalTensorMap: ...


def add(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
    alpha: object = 1,
    beta: object = 1,
) -> TensorMap | DiagonalTensorMap:
    if isinstance(t1, TensorMap):
        if not isinstance(t2, TensorMap):
            raise TypeError("add() requires a TensorMap")
        return t1.add(t2, alpha=alpha, beta=beta)
    if isinstance(t1, DiagonalTensorMap):
        if not isinstance(t2, DiagonalTensorMap):
            raise TypeError("add() requires a DiagonalTensorMap")
        return t1.add(t2, alpha=alpha, beta=beta)
    raise TypeError("add() requires a TensorMap or DiagonalTensorMap")


def inner(t1: TensorMap, t2: TensorMap) -> Array:
    _require_tensor_map(t1, "inner")
    return t1.inner(t2)


def dot(t1: TensorMap, t2: TensorMap) -> Array:
    _require_tensor_map(t1, "dot")
    return t1.dot(t2)


def norm(
    tensor: TensorMap | DiagonalTensorMap,
    p: SupportsFloat = 2,
) -> Array:
    _require_tensor_like(tensor, "norm")
    return tensor.norm(p=p)


def normalize(tensor: TensorMap, p: SupportsFloat = 2) -> TensorMap:
    _require_tensor_map(tensor, "normalize")
    return tensor.normalize(p=p)


def scalar(tensor: TensorMap) -> Array:
    _require_tensor_map(tensor, "scalar")
    return tensor.scalar()


@overload
def adjoint(tensor: TensorMap) -> TensorMap: ...


@overload
def adjoint(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def adjoint(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "adjoint")
    return tensor.adjoint()


@overload
def real(tensor: TensorMap) -> TensorMap: ...


@overload
def real(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def real(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "real")
    return tensor.real()


@overload
def imag(tensor: TensorMap) -> TensorMap: ...


@overload
def imag(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def imag(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "imag")
    return tensor.imag()


@overload
def complex(tensor: TensorMap) -> TensorMap: ...


@overload
def complex(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def complex(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    _require_tensor_like(tensor, "complex")
    return tensor.complex()


def tr(tensor: TensorMap) -> Array:
    _require_tensor_map(tensor, "tr")
    return tensor.tr()


def diag(tensor: TensorMap) -> SectorDict[Array]:
    _require_tensor_map(tensor, "diag")
    return tensor.diag()


def is_diagonal(tensor: TensorMap | DiagonalTensorMap) -> Array:
    _require_tensor_like(tensor, "is_diagonal")
    return tensor.is_diagonal()


@overload
def inverse(tensor: TensorMap) -> TensorMap: ...


@overload
def inverse(tensor: DiagonalTensorMap) -> DiagonalTensorMap: ...


def inverse(
    tensor: TensorMap | DiagonalTensorMap,
) -> TensorMap | DiagonalTensorMap:
    """Return the blockwise inverse with reversed domain and codomain."""
    _require_tensor_like(tensor, "inverse")
    if isinstance(tensor, DiagonalTensorMap):
        return tensor.inverse()

    result_space = _require_square_operator(tensor, "inverse")
    dtype = jnp.result_type(tensor.storage.data, 1.0)
    blocks = (
        (coupled, jnp.linalg.inv(jnp.asarray(block, dtype=dtype)))
        for coupled, block in tensor.blocks()
    )
    return TensorMap(result_space, pack_blocks(result_space, blocks, dtype=dtype))


@overload
def pseudoinverse(
    tensor: TensorMap,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> TensorMap: ...


@overload
def pseudoinverse(
    tensor: DiagonalTensorMap,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> DiagonalTensorMap: ...


def pseudoinverse(
    tensor: TensorMap | DiagonalTensorMap,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
) -> TensorMap | DiagonalTensorMap:
    """Return the blockwise Moore-Penrose pseudoinverse.

    Singular values at or below ``max(atol, rtol * max(singular_values))``
    are rejected. The default ``rtol`` is ten times the largest reduced block
    dimension times machine epsilon. The empty pseudoinverse is empty.
    """
    _require_tensor_like(tensor, "pseudoinverse")
    if isinstance(tensor, DiagonalTensorMap):
        return tensor.pseudoinverse(atol=atol, rtol=rtol)

    atol_value = nonnegative_tolerance(atol, "atol")
    rtol_value = (
        None if rtol is None else nonnegative_tolerance(rtol, "rtol")
    )
    result_space = hom(tensor.domain, tensor.codomain)
    dtype = jnp.result_type(tensor.storage.data, 1.0)
    real_dtype = jnp.abs(jnp.zeros((), dtype=dtype)).dtype
    decompositions: list[tuple[tuple[int, ...], Array, Array, Array]] = []
    largest = jnp.asarray(0, dtype=real_dtype)
    largest_block_dim = 0
    for coupled, block in tensor.blocks():
        block = jnp.asarray(block, dtype=dtype)
        u_block, singular_values, vh_block = jnp.linalg.svd(
            block,
            full_matrices=False,
        )
        decompositions.append((coupled, u_block, singular_values, vh_block))
        largest_block_dim = max(largest_block_dim, *block.shape)
        if singular_values.size > 0:
            largest = jnp.maximum(largest, jnp.max(singular_values))

    if rtol_value is None:
        rtol_value = default_pseudoinverse_rtol(real_dtype, largest_block_dim)
    cutoff = jnp.maximum(
        jnp.asarray(atol_value, dtype=real_dtype),
        jnp.asarray(rtol_value, dtype=real_dtype) * largest,
    )

    blocks: list[tuple[tuple[int, ...], Array]] = []
    for coupled, u_block, singular_values, vh_block in decompositions:
        accepted = singular_values > cutoff
        safe_values = jnp.where(
            accepted,
            singular_values,
            jnp.ones_like(singular_values),
        )
        inverse_values = jnp.where(
            accepted,
            jnp.ones_like(singular_values) / safe_values,
            jnp.zeros_like(singular_values),
        )
        block = (
            jnp.conj(vh_block.T) * inverse_values[None, :]
        ) @ jnp.conj(u_block.T)
        blocks.append((coupled, block))

    return TensorMap(
        result_space,
        pack_blocks(result_space, blocks, dtype=dtype),
    )


def left_solve(operator: TensorMap, rhs: TensorMap) -> TensorMap:
    """Return ``x`` satisfying ``operator @ x == rhs`` blockwise."""
    _require_tensor_map(operator, "left_solve")
    _require_tensor_map(rhs, "left_solve")
    result_space = _validate_left_solve(operator, rhs)
    dtype = jnp.result_type(operator.storage.data, rhs.storage.data, 1.0)
    operator_blocks = SectorDict(operator.blocks())
    rhs_blocks = SectorDict(rhs.blocks())
    blocks = (
        (
            coupled,
            jnp.linalg.solve(
                jnp.asarray(operator_blocks[coupled], dtype=dtype),
                jnp.asarray(block, dtype=dtype),
            ),
        )
        for coupled, block in rhs_blocks.items()
    )
    return TensorMap(
        result_space,
        pack_blocks(result_space, blocks, dtype=dtype),
    )


def right_solve(lhs: TensorMap, operator: TensorMap) -> TensorMap:
    """Return ``x`` satisfying ``x @ operator == lhs`` blockwise."""
    _require_tensor_map(lhs, "right_solve")
    _require_tensor_map(operator, "right_solve")
    result_space = _validate_right_solve(lhs, operator)
    dtype = jnp.result_type(lhs.storage.data, operator.storage.data, 1.0)
    lhs_blocks = SectorDict(lhs.blocks())
    operator_blocks = SectorDict(operator.blocks())
    blocks = (
        (
            coupled,
            jnp.linalg.solve(
                jnp.asarray(operator_blocks[coupled].T, dtype=dtype),
                jnp.asarray(block.T, dtype=dtype),
            ).T,
        )
        for coupled, block in lhs_blocks.items()
    )
    return TensorMap(
        result_space,
        pack_blocks(result_space, blocks, dtype=dtype),
    )


def is_isometric(
    tensor: TensorMap,
    *,
    side: Literal["left", "right"] = "left",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Array:
    """Return whether a left or right composition is the matching identity.

    The empty isometry is considered isometric.
    """
    _require_tensor_map(tensor, "is_isometric")
    if side not in ("left", "right"):
        raise ValueError("is_isometric() side must be 'left' or 'right'")
    atol_value, rtol_value = _predicate_tolerances(atol, rtol)

    from .constructors import identity

    if side == "left":
        if not is_monomorphic(fuse(tensor.domain), fuse(tensor.codomain)):
            return jnp.asarray(False)
        gram = tensor.adjoint() @ tensor
        expected = identity(tensor.domain, dtype=gram.dtype)
    else:
        if not is_monomorphic(fuse(tensor.codomain), fuse(tensor.domain)):
            return jnp.asarray(False)
        gram = tensor @ tensor.adjoint()
        expected = identity(tensor.codomain, dtype=gram.dtype)
    return allclose(gram, expected, atol=atol_value, rtol=rtol_value)


def is_unitary(
    tensor: TensorMap,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Array:
    """Return whether both compositions with the adjoint are identities.

    The empty unitary is considered unitary.
    """
    _require_tensor_map(tensor, "is_unitary")
    atol_value, rtol_value = _predicate_tolerances(atol, rtol)
    if not is_isomorphic(fuse(tensor.domain), fuse(tensor.codomain)):
        return jnp.asarray(False)

    from .constructors import identity

    tensor_adjoint = tensor.adjoint()
    right = allclose(
        tensor_adjoint @ tensor,
        identity(tensor.domain, dtype=tensor.dtype),
        atol=atol_value,
        rtol=rtol_value,
    )
    left = allclose(
        tensor @ tensor_adjoint,
        identity(tensor.codomain, dtype=tensor.dtype),
        atol=atol_value,
        rtol=rtol_value,
    )
    return jnp.logical_and(left, right)


def is_positive_definite(
    tensor: TensorMap,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Array:
    """Return whether a Hermitian endomorphism is strictly positive.

    Eigenvalues must exceed ``max(atol, rtol * max(abs(eigenvalues)))``.
    The empty endomorphism is considered positive definite.
    """
    _require_tensor_map(tensor, "is_positive_definite")
    atol_value, rtol_value = _predicate_tolerances(atol, rtol)
    if tensor.codomain != tensor.domain:
        raise ValueError(
            "is_positive_definite() requires an endomorphism with equal "
            "codomain and domain spaces",
        )

    hermitian = jnp.asarray(True)
    eigenvalue_blocks: list[Array] = []
    for _coupled, block in tensor.blocks():
        hermitian = jnp.logical_and(
            hermitian,
            jnp.allclose(
                block,
                jnp.conj(block.T),
                atol=atol_value,
                rtol=rtol_value,
            ),
        )
        if block.shape[0] > 0:
            eigenvalue_blocks.append(
                jnp.linalg.eigvalsh(
                    block,
                    UPLO="L",
                    symmetrize_input=False,
                ),
            )

    if not eigenvalue_blocks:
        return hermitian

    eigenvalues = jnp.concatenate(tuple(eigenvalue_blocks))
    cutoff = jnp.maximum(
        jnp.asarray(atol_value, dtype=eigenvalues.dtype),
        jnp.asarray(rtol_value, dtype=eigenvalues.dtype)
        * jnp.max(jnp.abs(eigenvalues)),
    )
    return jnp.logical_and(hermitian, jnp.min(eigenvalues) > cutoff)


def _compose(left: TensorMap, right: TensorMap) -> TensorMap:
    """Compose two TensorMaps blockwise over their shared middle space."""
    if left.space.domain != right.space.codomain:
        raise ValueError(
            "TensorMap spaces are not composable: "
            "left domain must equal right codomain",
        )

    result_space = hom(left.space.codomain, right.space.domain)
    dtype = jnp.result_type(left.storage.data, right.storage.data)
    left_blocks = SectorDict(left.blocks())
    right_blocks = SectorDict(right.blocks())
    result_blocks: list[tuple[tuple[int, ...], Array]] = []

    for coupled in get_sectorstructure(result_space).blocksectors:
        left_block = left_blocks.get(coupled)
        right_block = right_blocks.get(coupled)
        if left_block is not None and right_block is not None:
            result_blocks.append((coupled, left_block @ right_block))

    return TensorMap(
        result_space,
        pack_blocks(
            result_space,
            SectorDict(result_blocks),
            dtype=dtype,
        ),
    )


def tensor_product(left: TensorMap, right: TensorMap) -> TensorMap:
    """Return the disconnected tensor-network product of two tensor maps."""
    if not isinstance(left, TensorMap) or not isinstance(right, TensorMap):
        raise TypeError(
            "tensor_product() requires left and right to be TensorMap instances",
        )

    output = (
        tuple((0, axis) for axis in left.codomainind)
        + tuple((1, axis) for axis in right.codomainind),
        tuple((0, axis) for axis in left.domainind)
        + tuple((1, axis) for axis in right.domainind),
    )
    from ..operations.contractions.primitives import tensorcontract

    return tensorcontract(
        left,
        right,
        axes=((), ()),
        output=output,
    )


def equal(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
) -> Array:
    _require_tensor_like(t1, "equal")
    _require_tensor_like(t2, "equal")
    if t1.space != t2.space or t1.dtype != t2.dtype:
        return jnp.asarray(False)

    left, right = _comparison_storage(t1, t2)
    return jnp.array_equal(left, right)


def allclose(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
    *,
    rtol: float,
    atol: float,
) -> Array:
    _require_tensor_like(t1, "allclose")
    _require_tensor_like(t2, "allclose")
    if t1.space != t2.space:
        return jnp.asarray(False)

    left, right = _comparison_storage(t1, t2)
    return jnp.allclose(left, right, rtol=rtol, atol=atol)


def _require_square_operator(
    operator: TensorMap,
    function_name: str,
) -> _native.HomSpace:
    if not is_isomorphic(fuse(operator.codomain), fuse(operator.domain)):
        raise ValueError(
            f"{function_name}() requires isomorphic codomain and domain spaces",
        )
    return hom(operator.domain, operator.codomain)


def _validate_left_solve(
    operator: TensorMap,
    rhs: TensorMap,
) -> _native.HomSpace:
    _require_square_operator(operator, "left_solve")
    if rhs.codomain != operator.codomain:
        raise ValueError(
            "left_solve() requires rhs codomain to equal operator codomain",
        )
    return hom(operator.domain, rhs.domain)


def _validate_right_solve(
    lhs: TensorMap,
    operator: TensorMap,
) -> _native.HomSpace:
    _require_square_operator(operator, "right_solve")
    if lhs.domain != operator.domain:
        raise ValueError(
            "right_solve() requires lhs domain to equal operator domain",
        )
    return hom(lhs.codomain, operator.codomain)


def _predicate_tolerances(atol: float, rtol: float) -> tuple[float, float]:
    return (
        nonnegative_tolerance(atol, "atol"),
        nonnegative_tolerance(rtol, "rtol"),
    )


def _require_tensor_map(tensor: object, function_name: str) -> None:
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"{function_name}() requires a TensorMap")


def _require_tensor_like(tensor: object, function_name: str) -> None:
    if not isinstance(tensor, (TensorMap, DiagonalTensorMap)):
        raise TypeError(
            f"{function_name}() requires a TensorMap or DiagonalTensorMap",
        )


def _comparison_storage(
    t1: TensorMap | DiagonalTensorMap,
    t2: TensorMap | DiagonalTensorMap,
) -> tuple[Array, Array]:
    if isinstance(t1, DiagonalTensorMap) == isinstance(t2, DiagonalTensorMap):
        return t1.storage.data, t2.storage.data

    left = t1.to_tensor_map() if isinstance(t1, DiagonalTensorMap) else t1
    right = t2.to_tensor_map() if isinstance(t2, DiagonalTensorMap) else t2
    return left.storage.data, right.storage.data
