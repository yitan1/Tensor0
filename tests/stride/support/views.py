"""Shared views fixtures and references."""

from __future__ import annotations

import jax

from tensor0._stride import StridedView


def _pitched(data: jax.Array) -> StridedView:
    return StridedView(data, (2, 3), (1, 4), 2)


def _dense(data: jax.Array) -> StridedView:
    return StridedView.from_dense(data, (2, 3))
