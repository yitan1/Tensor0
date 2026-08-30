"""Private, sector-independent functional stride layer."""

from ._ffi import strided_copy
from ._materialize import materialize
from ._native_reduction import strided_reduce
from ._plan import (
    CompleteMode,
    StridedCopyPlan,
    StridedCopyRecord,
    StridedOutputInit,
    StridedReductionKind,
    StridedScalarKind,
    StridedWriteKind,
    build_strided_copy_plan,
)
from ._selected_scale import strided_scale
from ._update import strided_accumulate, strided_assign
from ._view import StridedView

__all__ = [
    "CompleteMode",
    "StridedCopyPlan",
    "StridedCopyRecord",
    "StridedOutputInit",
    "StridedReductionKind",
    "StridedScalarKind",
    "StridedView",
    "StridedWriteKind",
    "build_strided_copy_plan",
    "materialize",
    "strided_accumulate",
    "strided_assign",
    "strided_copy",
    "strided_reduce",
    "strided_scale",
]
