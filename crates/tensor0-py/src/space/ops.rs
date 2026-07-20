use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, GroupSpec, SU2Irrep, Sector, SectorSpec as CoreSectorSpec, Trivial,
    U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::{supremum_space as core_supremum_space, GradedSpace};

use crate::pyconv::core_err;
use crate::sector_type::PySectorSpec;

use super::graded::{GradedSpaceInner, PyElementarySpace};
use super::product::{ProductSpaceInner, PyProductSpace};

macro_rules! dispatch_binary_space_result {
    ($left:expr, $right:expr, $operation:path, $name:literal) => {
        match ($left, $right) {
            (GradedSpaceInner::Trivial(left), GradedSpaceInner::Trivial(right)) => {
                $operation(left, right).map(GradedSpaceInner::Trivial)
            }
            (GradedSpaceInner::U1Irrep(left), GradedSpaceInner::U1Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::U1Irrep)
            }
            (GradedSpaceInner::SU2Irrep(left), GradedSpaceInner::SU2Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::SU2Irrep)
            }
            (GradedSpaceInner::FermionParity(left), GradedSpaceInner::FermionParity(right)) => {
                $operation(left, right).map(GradedSpaceInner::FermionParity)
            }
            (GradedSpaceInner::Z2Irrep(left), GradedSpaceInner::Z2Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::Z2Irrep)
            }
            (GradedSpaceInner::Z3Irrep(left), GradedSpaceInner::Z3Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::Z3Irrep)
            }
            (GradedSpaceInner::Z4Irrep(left), GradedSpaceInner::Z4Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::Z4Irrep)
            }
            (
                GradedSpaceInner::U1IrrepFermionParity(left),
                GradedSpaceInner::U1IrrepFermionParity(right),
            ) => $operation(left, right).map(GradedSpaceInner::U1IrrepFermionParity),
            (
                GradedSpaceInner::FermionParityU1Irrep(left),
                GradedSpaceInner::FermionParityU1Irrep(right),
            ) => $operation(left, right).map(GradedSpaceInner::FermionParityU1Irrep),
            (GradedSpaceInner::U1SU2Irrep(left), GradedSpaceInner::U1SU2Irrep(right)) => {
                $operation(left, right).map(GradedSpaceInner::U1SU2Irrep)
            }
            (
                GradedSpaceInner::FermionParitySU2Irrep(left),
                GradedSpaceInner::FermionParitySU2Irrep(right),
            ) => $operation(left, right).map(GradedSpaceInner::FermionParitySU2Irrep),
            (
                GradedSpaceInner::FermionParityU1SU2Irrep(left),
                GradedSpaceInner::FermionParityU1SU2Irrep(right),
            ) => $operation(left, right).map(GradedSpaceInner::FermionParityU1SU2Irrep),
            _ => {
                return Err(PyValueError::new_err(concat!(
                    $name,
                    " spaces must have the same sector family"
                )))
            }
        }
        .map_err(core_err)
    };
}

macro_rules! dispatch_binary_space_predicate {
    ($left:expr, $right:expr, $method:ident) => {
        match ($left, $right) {
            (GradedSpaceInner::Trivial(left), GradedSpaceInner::Trivial(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::U1Irrep(left), GradedSpaceInner::U1Irrep(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::SU2Irrep(left), GradedSpaceInner::SU2Irrep(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::FermionParity(left), GradedSpaceInner::FermionParity(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::Z2Irrep(left), GradedSpaceInner::Z2Irrep(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::Z3Irrep(left), GradedSpaceInner::Z3Irrep(right)) => {
                left.$method(right)
            }
            (GradedSpaceInner::Z4Irrep(left), GradedSpaceInner::Z4Irrep(right)) => {
                left.$method(right)
            }
            (
                GradedSpaceInner::U1IrrepFermionParity(left),
                GradedSpaceInner::U1IrrepFermionParity(right),
            ) => left.$method(right),
            (
                GradedSpaceInner::FermionParityU1Irrep(left),
                GradedSpaceInner::FermionParityU1Irrep(right),
            ) => left.$method(right),
            (GradedSpaceInner::U1SU2Irrep(left), GradedSpaceInner::U1SU2Irrep(right)) => {
                left.$method(right)
            }
            (
                GradedSpaceInner::FermionParitySU2Irrep(left),
                GradedSpaceInner::FermionParitySU2Irrep(right),
            ) => left.$method(right),
            (
                GradedSpaceInner::FermionParityU1SU2Irrep(left),
                GradedSpaceInner::FermionParityU1SU2Irrep(right),
            ) => left.$method(right),
            _ => false,
        }
    };
}

#[pyfunction]
pub(crate) fn dim(space: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(space) = space.extract::<PyRef<'_, PyElementarySpace>>() {
        return Ok(elementary_dim(&space.inner));
    }
    if let Ok(space) = space.extract::<PyRef<'_, PyProductSpace>>() {
        return Ok(product_dim(space.inner()));
    }
    Err(PyTypeError::new_err(
        "dim() requires an ElementarySpace or ProductSpace",
    ))
}

#[pyfunction]
pub(crate) fn reduced_dim(space: PyRef<'_, PyElementarySpace>) -> usize {
    elementary_reduced_dim(&space.inner)
}

#[pyfunction(signature = (sector_spec, is_dual=false))]
pub(crate) fn unit_space(
    sector_spec: PyRef<'_, PySectorSpec>,
    is_dual: bool,
) -> PyResult<PyElementarySpace> {
    canonical_space(&sector_spec.inner, is_dual, true).map(|inner| PyElementarySpace { inner })
}

#[pyfunction(signature = (sector_spec, is_dual=false))]
pub(crate) fn zero_space(
    sector_spec: PyRef<'_, PySectorSpec>,
    is_dual: bool,
) -> PyResult<PyElementarySpace> {
    canonical_space(&sector_spec.inner, is_dual, false).map(|inner| PyElementarySpace { inner })
}

#[pyfunction]
pub(crate) fn supremum_space(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> PyResult<PyElementarySpace> {
    dispatch_binary_space_result!(&left.inner, &right.inner, core_supremum_space, "supremum")
        .map(|inner| PyElementarySpace { inner })
}

#[pyfunction]
pub(crate) fn direct_sum(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> PyResult<PyElementarySpace> {
    dispatch_binary_space_result!(&left.inner, &right.inner, core_direct_sum, "direct sum")
        .map(|inner| PyElementarySpace { inner })
}

#[pyfunction]
pub(crate) fn is_isomorphic(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> bool {
    dispatch_binary_space_predicate!(&left.inner, &right.inner, is_isomorphic)
}

#[pyfunction]
pub(crate) fn is_monomorphic(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> bool {
    dispatch_binary_space_predicate!(&left.inner, &right.inner, is_monomorphic)
}

#[pyfunction]
pub(crate) fn is_epimorphic(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> bool {
    dispatch_binary_space_predicate!(&left.inner, &right.inner, is_epimorphic)
}

fn elementary_dim(space: &GradedSpaceInner) -> usize {
    match space {
        GradedSpaceInner::Trivial(space) => space.dim(),
        GradedSpaceInner::U1Irrep(space) => space.dim(),
        GradedSpaceInner::SU2Irrep(space) => space.dim(),
        GradedSpaceInner::FermionParity(space) => space.dim(),
        GradedSpaceInner::Z2Irrep(space) => space.dim(),
        GradedSpaceInner::Z3Irrep(space) => space.dim(),
        GradedSpaceInner::Z4Irrep(space) => space.dim(),
        GradedSpaceInner::U1IrrepFermionParity(space) => space.dim(),
        GradedSpaceInner::FermionParityU1Irrep(space) => space.dim(),
        GradedSpaceInner::U1SU2Irrep(space) => space.dim(),
        GradedSpaceInner::FermionParitySU2Irrep(space) => space.dim(),
        GradedSpaceInner::FermionParityU1SU2Irrep(space) => space.dim(),
    }
}

fn elementary_reduced_dim(space: &GradedSpaceInner) -> usize {
    match space {
        GradedSpaceInner::Trivial(space) => space.reduced_dim(),
        GradedSpaceInner::U1Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::SU2Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::FermionParity(space) => space.reduced_dim(),
        GradedSpaceInner::Z2Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::Z3Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::Z4Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::U1IrrepFermionParity(space) => space.reduced_dim(),
        GradedSpaceInner::FermionParityU1Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::U1SU2Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::FermionParitySU2Irrep(space) => space.reduced_dim(),
        GradedSpaceInner::FermionParityU1SU2Irrep(space) => space.reduced_dim(),
    }
}

fn product_dim(space: &ProductSpaceInner) -> usize {
    match space {
        ProductSpaceInner::Trivial(space) => space.dim(),
        ProductSpaceInner::U1Irrep(space) => space.dim(),
        ProductSpaceInner::SU2Irrep(space) => space.dim(),
        ProductSpaceInner::FermionParity(space) => space.dim(),
        ProductSpaceInner::Z2Irrep(space) => space.dim(),
        ProductSpaceInner::Z3Irrep(space) => space.dim(),
        ProductSpaceInner::Z4Irrep(space) => space.dim(),
        ProductSpaceInner::U1IrrepFermionParity(space) => space.dim(),
        ProductSpaceInner::FermionParityU1Irrep(space) => space.dim(),
        ProductSpaceInner::U1SU2Irrep(space) => space.dim(),
        ProductSpaceInner::FermionParitySU2Irrep(space) => space.dim(),
        ProductSpaceInner::FermionParityU1SU2Irrep(space) => space.dim(),
    }
}

fn core_direct_sum<I: Sector>(
    left: &GradedSpace<I>,
    right: &GradedSpace<I>,
) -> tensor0_core::error::Result<GradedSpace<I>> {
    left.direct_sum(right)
}

fn canonical_space(spec: &CoreSectorSpec, is_dual: bool, unit: bool) -> PyResult<GradedSpaceInner> {
    match spec.clone().canonicalize().map_err(core_err)? {
        CoreSectorSpec::Trivial => Ok(canonical_space_variant::<Trivial>(
            is_dual,
            unit,
            GradedSpaceInner::Trivial,
        )),
        CoreSectorSpec::Irrep {
            group: GroupSpec::U1,
        } => Ok(canonical_space_variant::<U1Irrep>(
            is_dual,
            unit,
            GradedSpaceInner::U1Irrep,
        )),
        CoreSectorSpec::Irrep {
            group: GroupSpec::SU2,
        } => Ok(canonical_space_variant::<SU2Irrep>(
            is_dual,
            unit,
            GradedSpaceInner::SU2Irrep,
        )),
        CoreSectorSpec::FermionParity => Ok(canonical_space_variant::<FermionParity>(
            is_dual,
            unit,
            GradedSpaceInner::FermionParity,
        )),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 2 },
        } => Ok(canonical_space_variant::<Z2Irrep>(
            is_dual,
            unit,
            GradedSpaceInner::Z2Irrep,
        )),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 3 },
        } => Ok(canonical_space_variant::<Z3Irrep>(
            is_dual,
            unit,
            GradedSpaceInner::Z3Irrep,
        )),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 4 },
        } => Ok(canonical_space_variant::<Z4Irrep>(
            is_dual,
            unit,
            GradedSpaceInner::Z4Irrep,
        )),
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::fermion_parity()] =>
        {
            Ok(canonical_space_variant::<FermionNumber>(
                is_dual,
                unit,
                GradedSpaceInner::U1IrrepFermionParity,
            ))
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::u1()] =>
        {
            Ok(canonical_space_variant::<FermionParityU1Irrep>(
                is_dual,
                unit,
                GradedSpaceInner::FermionParityU1Irrep,
            ))
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::su2()] =>
        {
            Ok(canonical_space_variant::<U1SU2Irrep>(
                is_dual,
                unit,
                GradedSpaceInner::U1SU2Irrep,
            ))
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::su2()] =>
        {
            Ok(canonical_space_variant::<FermionParitySU2Irrep>(
                is_dual,
                unit,
                GradedSpaceInner::FermionParitySU2Irrep,
            ))
        }
        CoreSectorSpec::Product { components }
            if components
                == vec![
                    CoreSectorSpec::fermion_parity(),
                    CoreSectorSpec::u1(),
                    CoreSectorSpec::su2(),
                ] =>
        {
            Ok(canonical_space_variant::<FermionParityU1SU2Irrep>(
                is_dual,
                unit,
                GradedSpaceInner::FermionParityU1SU2Irrep,
            ))
        }
        _ => Err(PyValueError::new_err("unsupported sector spec")),
    }
}

fn canonical_space_variant<I: Sector>(
    is_dual: bool,
    unit: bool,
    wrap: fn(GradedSpace<I>) -> GradedSpaceInner,
) -> GradedSpaceInner {
    let space = if unit {
        GradedSpace::<I>::unit()
    } else {
        GradedSpace::<I>::zero(false)
    };
    wrap(if is_dual { space.dual() } else { space })
}
