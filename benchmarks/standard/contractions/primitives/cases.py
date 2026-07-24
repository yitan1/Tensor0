from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

from tensor0 import (
    ComplexSpace,
    FermionParity,
    SU2Irrep,
    TensorMap,
    U1Irrep,
    hom,
    space,
    tensorcontract,
    tensortrace,
)

from ..._inputs import dense_tensor, packed_tensor
from ..._runner import Operation
from ..._specs import (
    EAGER,
    EXPLICIT,
    FULL,
    JIT_CACHED,
    JIT_COMPILE,
    QUICK,
    PreparedOperation,
    ScenarioProfile,
    ScenarioSpec,
    WorkloadSpec,
    prepared_from_operation,
)
def _trace_u1_partial() -> Operation:
    open_space = space(U1Irrep, {0: 8, 1: 4})
    traced = space(U1Irrep, {0: 6, 1: 3})
    tensor = packed_tensor(
        hom((open_space, traced), (open_space, traced)),
        scale=0.01,
    )

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run


def _trace_su2_partial() -> Operation:
    half = space(SU2Irrep, {1: 2})
    tensor = packed_tensor(hom((half, half), (half, half)), scale=0.01)

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run


def _trace_fermion_full() -> Operation:
    factor = space(FermionParity, {0: 16, 1: 16})
    tensor = packed_tensor(hom((factor,), (factor,)), scale=0.01)

    def run() -> object:
        return tensortrace(
            tensor,
            axes=((0,), (1,)),
            output=((), ()),
        )

    return run


def _contract_u1_partial(*, dtype: str = "float64") -> Operation:
    a = space(U1Irrep, {0: 8, 1: 4})
    x = space(U1Irrep, {0: 6, 1: 3})
    b = space(U1Irrep, {0: 7, 1: 5})
    c = space(U1Irrep, {0: 4, 1: 2})
    d = space(U1Irrep, {0: 5, 1: 3})
    left = packed_tensor(hom((a, x), (c,)), dtype=dtype, scale=0.01)
    right = packed_tensor(
        hom((x.dual(), b), (d,)),
        dtype=dtype,
        scale=0.01,
    )

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0), (1, 1)), ((0, 2), (1, 2))),
        )

    return run


def _contract_su2_fusion_basis() -> Operation:
    half = space(SU2Irrep, {1: 1})
    left = packed_tensor(hom((half, half, half), (half,)), scale=0.01)
    right = packed_tensor(
        hom((half.dual(), half, half), (half,)),
        scale=0.01,
    )

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((2,), (0,)),
            output=(
                ((0, 0), (0, 1), (1, 1), (1, 2)),
                ((0, 3), (1, 3)),
            ),
        )

    return run


def _contract_fermion_twist() -> Operation:
    odd = space(FermionParity, {1: 16})
    odd_dual = odd.dual()
    left = packed_tensor(hom((odd,), (odd_dual,)), scale=0.01)
    right = packed_tensor(hom((odd_dual,), (odd,)), scale=0.01)

    def run() -> object:
        return tensorcontract(
            left,
            right,
            axes=((1,), (0,)),
            output=(((0, 0),), ((1, 1),)),
        )

    return run


def _trivial_trace_prepared(
    *,
    dtype: str,
    size_label: str,
) -> PreparedOperation:
    dimensions = {
        "small": (8, 4, 6),
        "medium": (24, 12, 20),
    }[size_label]
    output_dim, traced_dim, input_dim = dimensions
    traced = ComplexSpace(traced_dim)
    target = hom(
        (ComplexSpace(output_dim), traced),
        (ComplexSpace(input_dim), traced),
    )
    tensor = dense_tensor(target, dtype=dtype)

    def run(value: TensorMap) -> object:
        return tensortrace(
            value,
            axes=((1,), (3,)),
            output=((0,), (2,)),
        )

    return run, (tensor,)


def _trivial_contract_prepared(
    *,
    dtype: str,
    size_label: str,
) -> PreparedOperation:
    dimensions = {
        "small": (6, 4, 7, 5, 3),
        "medium": (16, 12, 18, 10, 8),
    }[size_label]
    a_dim, contracted_dim, b_dim, c_dim, d_dim = dimensions
    contracted = ComplexSpace(contracted_dim)
    left = dense_tensor(
        hom(
            (ComplexSpace(a_dim), contracted),
            (ComplexSpace(c_dim),),
        ),
        dtype=dtype,
    )
    right = dense_tensor(
        hom(
            (contracted.dual(), ComplexSpace(b_dim)),
            (ComplexSpace(d_dim),),
        ),
        dtype=dtype,
    )

    def run(left_value: TensorMap, right_value: TensorMap) -> object:
        return tensorcontract(
            left_value,
            right_value,
            axes=((1,), (0,)),
            output=(((0, 0), (1, 1)), ((0, 2), (1, 2))),
        )

    return run, (left, right)


_EAGER_NO_SUFFIX = replace(EAGER, id_suffix="")


def _workload(
    id: str,
    description: str,
    dtype: str,
    size_label: str,
    factory: Callable[[], Operation],
) -> WorkloadSpec:
    return WorkloadSpec(
        id,
        "contractions",
        description,
        dtype,
        size_label,
        prepared_from_operation(factory),
    )


def _spec(
    workload: WorkloadSpec,
    profile: ScenarioProfile,
) -> ScenarioSpec:
    return ScenarioSpec(workload, _EAGER_NO_SUFFIX, profile)


_STANDARD_PRIMITIVE_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    _spec(
        _workload(
            "trace.u1.partial",
            "U1 partial tensor trace",
            "float64",
            "small",
            _trace_u1_partial,
        ),
        QUICK,
    ),
    _spec(
        _workload(
            "contract.u1.partial",
            "U1 partial binary contraction",
            "float64",
            "small",
            _contract_u1_partial,
        ),
        QUICK,
    ),
    _spec(
        _workload(
            "trace.su2.partial",
            "SU2 partial tensor trace",
            "float64",
            "small",
            _trace_su2_partial,
        ),
        QUICK,
    ),
    _spec(
        _workload(
            "trace.fermion.full",
            "FermionParity full tensor trace",
            "float64",
            "small",
            _trace_fermion_full,
        ),
        FULL,
    ),
    _spec(
        _workload(
            "contract.su2.fusion_basis",
            "SU2 fusion-basis binary contraction",
            "float64",
            "small",
            _contract_su2_fusion_basis,
        ),
        FULL,
    ),
    _spec(
        _workload(
            "contract.fermion.twist",
            "FermionParity contraction with right twist",
            "float64",
            "small",
            _contract_fermion_twist,
        ),
        FULL,
    ),
    _spec(
        _workload(
            "contract.u1.partial.complex",
            "Complex U1 partial binary contraction",
            "complex128",
            "medium",
            partial(_contract_u1_partial, dtype="complex128"),
        ),
        EXPLICIT,
    ),
)

_TRIVIAL_TRACE_SMALL = WorkloadSpec(
    "trace.trivial.partial.rank4.float64.small",
    "contractions",
    "Trivial partial trace with source shape (8, 4, 6, 4)",
    "float64",
    "small",
    partial(
        _trivial_trace_prepared,
        dtype="float64",
        size_label="small",
    ),
)
_TRIVIAL_TRACE_MEDIUM = WorkloadSpec(
    "trace.trivial.partial.rank4.complex128.medium",
    "contractions",
    "Trivial partial trace with source shape (24, 12, 20, 12)",
    "complex128",
    "medium",
    partial(
        _trivial_trace_prepared,
        dtype="complex128",
        size_label="medium",
    ),
)
_TRIVIAL_CONTRACT_SMALL = WorkloadSpec(
    "contract.trivial.partial.float64.small",
    "contractions",
    "Trivial partial contraction with dimensions (6, 4, 7, 5, 3)",
    "float64",
    "small",
    partial(
        _trivial_contract_prepared,
        dtype="float64",
        size_label="small",
    ),
)
_TRIVIAL_CONTRACT_MEDIUM = WorkloadSpec(
    "contract.trivial.partial.complex128.medium",
    "contractions",
    "Trivial partial contraction with dimensions (16, 12, 18, 10, 8)",
    "complex128",
    "medium",
    partial(
        _trivial_contract_prepared,
        dtype="complex128",
        size_label="medium",
    ),
)
TRIVIAL_PRIMITIVE_SCENARIO_SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(_TRIVIAL_TRACE_SMALL, EAGER, QUICK),
    ScenarioSpec(_TRIVIAL_TRACE_SMALL, JIT_COMPILE, FULL),
    ScenarioSpec(_TRIVIAL_TRACE_SMALL, JIT_CACHED, FULL),
    ScenarioSpec(_TRIVIAL_TRACE_MEDIUM, EAGER, FULL),
    ScenarioSpec(_TRIVIAL_CONTRACT_SMALL, EAGER, QUICK),
    ScenarioSpec(_TRIVIAL_CONTRACT_SMALL, JIT_COMPILE, FULL),
    ScenarioSpec(_TRIVIAL_CONTRACT_SMALL, JIT_CACHED, FULL),
    ScenarioSpec(_TRIVIAL_CONTRACT_MEDIUM, EAGER, FULL),
)
PRIMITIVE_SCENARIO_SPECS = (
    _STANDARD_PRIMITIVE_SCENARIO_SPECS
    + TRIVIAL_PRIMITIVE_SCENARIO_SPECS
)
