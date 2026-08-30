"""Stable diagnostics shared by Tensor0 stride execution routes."""

from __future__ import annotations

from typing import Never


NO_ROUTE_PREFIX = "tensor0-stride no eligible route:"


def raise_no_eligible_route(reasons: tuple[str, ...]) -> Never:
    """Raise the stable production routing diagnostic."""

    raise RuntimeError(f"{NO_ROUTE_PREFIX} {', '.join(reasons)}")


__all__ = ["NO_ROUTE_PREFIX", "raise_no_eligible_route"]
