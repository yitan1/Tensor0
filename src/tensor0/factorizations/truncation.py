from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import jax
from jax import Array
import jax.numpy as jnp

from .. import _native
from ..tensor.sector_vector import SectorVector

SectorKey = tuple[int, ...]
KeepIndices = dict[SectorKey, tuple[int, ...]]


class _TruncationStrategy:
    def __and__(self, other: object) -> _CombinedTruncation:
        return _CombinedTruncation("intersection", self, _ensure_strategy(other))

    def __or__(self, other: object) -> _CombinedTruncation:
        return _CombinedTruncation("union", self, _ensure_strategy(other))


@dataclass(frozen=True)
class NoTruncation(_TruncationStrategy):
    pass


@dataclass(frozen=True)
class TruncationByRank(_TruncationStrategy):
    howmany: int
    by: Callable[[float], float]
    rev: bool


@dataclass(frozen=True)
class TruncationByValue(_TruncationStrategy):
    atol: float
    rtol: float
    by: Callable[[float], float]
    keep_below: bool


@dataclass(frozen=True)
class TruncationByError(_TruncationStrategy):
    atol: float
    rtol: float


@dataclass(frozen=True)
class TruncationSpace(_TruncationStrategy):
    space: _native.ElementarySpace
    by: Callable[[float], float]
    rev: bool


@dataclass(frozen=True)
class _CombinedTruncation(_TruncationStrategy):
    op: str
    left: _TruncationStrategy
    right: _TruncationStrategy


def notrunc() -> NoTruncation:
    return NoTruncation()


def truncrank(
    howmany: int,
    *,
    by: Callable[[float], float] = abs,
    rev: bool = True,
) -> TruncationByRank:
    if isinstance(howmany, bool) or not isinstance(howmany, int):
        raise TypeError("truncrank() requires an integer rank")
    if howmany < 0:
        raise ValueError("truncrank() rank must be non-negative")
    _check_callable_by(by)
    return TruncationByRank(howmany, by, rev)


def trunctol(
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    by: Callable[[float], float] = abs,
    keep_below: bool = False,
) -> TruncationByValue:
    if atol < 0 or rtol < 0:
        raise ValueError("trunctol() tolerances must be non-negative")
    _check_callable_by(by)
    return TruncationByValue(float(atol), float(rtol), by, keep_below)


def truncerror(*, atol: float = 0.0, rtol: float = 0.0) -> TruncationByError:
    if atol < 0 or rtol < 0:
        raise ValueError("truncerror() tolerances must be non-negative")
    return TruncationByError(float(atol), float(rtol))


def truncspace(
    space: _native.ElementarySpace,
    *,
    by: Callable[[float], float] = abs,
    rev: bool = True,
) -> TruncationSpace:
    if not isinstance(space, _native.ElementarySpace):
        raise TypeError("truncspace() requires an ElementarySpace")
    if space.is_dual:
        raise ValueError("truncation space must not be dual")
    _check_callable_by(by)
    return TruncationSpace(space, by, rev)


def _check_callable_by(by: object) -> None:
    if not callable(by):
        raise TypeError("by must be callable")


def _ensure_strategy(value: object) -> _TruncationStrategy:
    if isinstance(value, _TruncationStrategy):
        return value
    raise TypeError("truncation strategy expected")


def _find_truncated_indices(
    values: SectorVector,
    strategy: _TruncationStrategy,
) -> KeepIndices:
    strategy = _ensure_strategy(strategy)
    if isinstance(strategy, NoTruncation):
        return {sector: tuple(range(block.shape[0])) for sector, block in values.blocks()}
    if isinstance(strategy, TruncationByRank):
        return _rank_indices(values, strategy)
    if isinstance(strategy, TruncationByValue):
        return _value_indices(values, strategy)
    if isinstance(strategy, TruncationByError):
        return _error_indices(values, strategy)
    if isinstance(strategy, TruncationSpace):
        return _space_indices(values, strategy)
    if isinstance(strategy, _CombinedTruncation):
        left = _find_truncated_indices(values, strategy.left)
        right = _find_truncated_indices(values, strategy.right)
        return _combine_indices(values, left, right, strategy.op)
    raise TypeError("unsupported truncation strategy")


def _rank_indices(values: SectorVector, strategy: TruncationByRank) -> KeepIndices:
    result = _empty_indices(values)
    used = 0
    for sector, index, _value, weight in _sorted_entries(values, strategy.by, strategy.rev):
        if used + weight > strategy.howmany:
            continue
        result[sector].append(index)
        used += weight
    return _freeze_indices(result)


def _value_indices(values: SectorVector, strategy: TruncationByValue) -> KeepIndices:
    cutoff = max(strategy.atol, _weighted_norm(values) * strategy.rtol)
    result = _empty_indices(values)
    for sector, index, value, _weight in _entries(values):
        score = strategy.by(value)
        keep = score <= cutoff if strategy.keep_below else score >= cutoff
        if keep:
            result[sector].append(index)
    return _freeze_indices(result)


def _error_indices(values: SectorVector, strategy: TruncationByError) -> KeepIndices:
    budget = max(strategy.atol, _weighted_norm(values) * strategy.rtol)
    budget_squared = budget * budget
    discarded_squared = 0.0
    keep = _all_index_sets(values)
    for sector, index, value, weight in _sorted_entries(values, abs, False):
        contribution = value * value * weight
        if discarded_squared + contribution > budget_squared:
            break
        keep[sector].remove(index)
        discarded_squared += contribution
    return _freeze_indices({sector: sorted(indices) for sector, indices in keep.items()})


def _space_indices(values: SectorVector, strategy: TruncationSpace) -> KeepIndices:
    if values.sector_type != strategy.space.sector_spec:
        raise ValueError("truncation space sector family must match singular values")
    limits = {sector: dim for sector, dim in strategy.space.sectors}
    result = _empty_indices(values)
    for sector, block in values.blocks():
        limit = min(int(block.shape[0]), limits.get(sector, 0))
        candidates = [
            (index, float(abs(raw_value)))
            for index, raw_value in enumerate(jax.device_get(block))
        ]
        candidates.sort(key=lambda item: strategy.by(item[1]), reverse=strategy.rev)
        selected = sorted(index for index, _value in candidates[:limit])
        result[sector].extend(selected)
    return _freeze_indices(result)


def _combine_indices(
    values: SectorVector,
    left: KeepIndices,
    right: KeepIndices,
    op: str,
) -> KeepIndices:
    result = _empty_indices(values)
    for sector, block in values.blocks():
        left_set = set(left.get(sector, ()))
        right_set = set(right.get(sector, ()))
        if op == "intersection":
            selected = left_set & right_set
        elif op == "union":
            selected = left_set | right_set
        else:
            raise ValueError(f"unknown truncation combination {op!r}")
        result[sector].extend(index for index in range(block.shape[0]) if index in selected)
    return _freeze_indices(result)


def _truncation_error(values: SectorVector, indices: KeepIndices) -> Array:
    keep = {sector: set(index_tuple) for sector, index_tuple in indices.items()}
    error_squared = 0.0
    for sector, index, value, weight in _entries(values):
        if index not in keep.get(sector, set()):
            error_squared += value * value * weight
    dtype = getattr(values.storage.data, "dtype", None)
    return jnp.asarray(math.sqrt(error_squared), dtype=dtype)


def _weighted_norm(values: SectorVector) -> float:
    total = 0.0
    for _sector, _index, value, weight in _entries(values):
        total += value * value * weight
    return math.sqrt(total)


def _entries(values: SectorVector) -> list[tuple[SectorKey, int, float, int]]:
    entries: list[tuple[SectorKey, int, float, int]] = []
    for sector, block in values.blocks():
        weight = values.sector_type.quantum_dim(sector)
        for index, raw_value in enumerate(jax.device_get(block)):
            entries.append((sector, index, float(abs(raw_value)), weight))
    return entries


def _sorted_entries(
    values: SectorVector,
    by: Callable[[float], float],
    rev: bool,
) -> list[tuple[SectorKey, int, float, int]]:
    entries = _entries(values)
    entries.sort(key=lambda entry: by(entry[2]), reverse=rev)
    return entries


def _empty_indices(values: SectorVector) -> dict[SectorKey, list[int]]:
    return {sector: [] for sector, _block in values.blocks()}


def _all_index_sets(values: SectorVector) -> dict[SectorKey, set[int]]:
    return {sector: set(range(block.shape[0])) for sector, block in values.blocks()}


def _freeze_indices(indices: dict[SectorKey, list[int]]) -> KeepIndices:
    return {sector: tuple(index_list) for sector, index_list in indices.items()}
