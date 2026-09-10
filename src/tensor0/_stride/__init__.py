"""Private, sector-independent functional view algebra.

Scale, addition, and internal assignment and accumulation share one functional
update core with zero/one short-circuit semantics and explicit AD rules.
Execution plans and multi-record execution remain private implementation
details; they are not exported by this facade.
"""

from ._algebra import add, dotc, dotu, reduce_sum, scale
from ._ops._materialize import materialize
from ._view import StridedView
from ._native import (
    disable_threads,
    enable_threads,
    get_num_threads,
    set_num_threads,
)

__all__ = [
    "StridedView",
    "add",
    "disable_threads",
    "dotc",
    "dotu",
    "enable_threads",
    "get_num_threads",
    "materialize",
    "reduce_sum",
    "scale",
    "set_num_threads",
]
