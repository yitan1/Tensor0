use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, GroupSpec, SU2Irrep, Sector, SectorSpec as CoreSectorSpec, Trivial,
    U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::{infimum_space as core_infimum_space, ElementarySpaceSpec, GradedSpace};

use crate::pyconv::{core_err, parse_sector_dims, py_hash, sectors_py};
use crate::sector_type::PySectorSpec;

use super::static_key::space_static_key;

#[pyclass(name = "ElementarySpace", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyElementarySpace {
    pub(super) inner: GradedSpaceInner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum GradedSpaceInner {
    Trivial(GradedSpace<Trivial>),
    U1Irrep(GradedSpace<U1Irrep>),
    SU2Irrep(GradedSpace<SU2Irrep>),
    FermionParity(GradedSpace<FermionParity>),
    Z2Irrep(GradedSpace<Z2Irrep>),
    Z3Irrep(GradedSpace<Z3Irrep>),
    Z4Irrep(GradedSpace<Z4Irrep>),
    U1IrrepFermionParity(GradedSpace<FermionNumber>),
    FermionParityU1Irrep(GradedSpace<FermionParityU1Irrep>),
    U1SU2Irrep(GradedSpace<U1SU2Irrep>),
    FermionParitySU2Irrep(GradedSpace<FermionParitySU2Irrep>),
    FermionParityU1SU2Irrep(GradedSpace<FermionParityU1SU2Irrep>),
}

#[pyfunction(signature = (sector_spec, sectors, is_dual=false))]
pub(crate) fn make_space(
    sector_spec: PyRef<'_, PySectorSpec>,
    sectors: &Bound<'_, PyAny>,
    is_dual: bool,
) -> PyResult<PyElementarySpace> {
    let spec = sector_spec.inner.clone().canonicalize().map_err(core_err)?;
    Ok(PyElementarySpace {
        inner: graded_space_from_spec(spec, sectors, is_dual)?,
    })
}

#[pyfunction]
pub(crate) fn infimum_space(
    left: PyRef<'_, PyElementarySpace>,
    right: PyRef<'_, PyElementarySpace>,
) -> PyResult<PyElementarySpace> {
    GradedSpaceInner::infimum(&left.inner, &right.inner).map(|inner| PyElementarySpace { inner })
}

#[pymethods]
impl PyElementarySpace {
    #[getter]
    fn sector_spec(&self) -> PySectorSpec {
        PySectorSpec {
            inner: self.inner.sector_spec(),
        }
    }

    #[getter]
    fn sectors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.sectors_py(py)
    }

    #[getter]
    fn is_dual(&self) -> bool {
        self.inner.is_dual()
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        space_static_key(py, &self.inner)
    }

    fn dual(&self) -> PyElementarySpace {
        PyElementarySpace {
            inner: self.inner.dual(),
        }
    }

    fn flip(&self) -> PyElementarySpace {
        PyElementarySpace {
            inner: self.inner.flip(),
        }
    }

    fn __eq__(&self, other: PyRef<'_, PyElementarySpace>) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        py_hash("ElementarySpace", &self.inner.to_spec())
    }
}

impl GradedSpaceInner {
    pub(super) fn sector_spec(&self) -> CoreSectorSpec {
        match self {
            GradedSpaceInner::Trivial(_) => Trivial::sector_spec(),
            GradedSpaceInner::U1Irrep(_) => U1Irrep::sector_spec(),
            GradedSpaceInner::SU2Irrep(_) => SU2Irrep::sector_spec(),
            GradedSpaceInner::FermionParity(_) => FermionParity::sector_spec(),
            GradedSpaceInner::Z2Irrep(_) => Z2Irrep::sector_spec(),
            GradedSpaceInner::Z3Irrep(_) => Z3Irrep::sector_spec(),
            GradedSpaceInner::Z4Irrep(_) => Z4Irrep::sector_spec(),
            GradedSpaceInner::U1IrrepFermionParity(_) => FermionNumber::sector_spec(),
            GradedSpaceInner::FermionParityU1Irrep(_) => FermionParityU1Irrep::sector_spec(),
            GradedSpaceInner::U1SU2Irrep(_) => U1SU2Irrep::sector_spec(),
            GradedSpaceInner::FermionParitySU2Irrep(_) => FermionParitySU2Irrep::sector_spec(),
            GradedSpaceInner::FermionParityU1SU2Irrep(_) => FermionParityU1SU2Irrep::sector_spec(),
        }
    }

    pub(super) fn sectors_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            GradedSpaceInner::Trivial(space) => sectors_py(py, space),
            GradedSpaceInner::U1Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::SU2Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::FermionParity(space) => sectors_py(py, space),
            GradedSpaceInner::Z2Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::Z3Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::Z4Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::U1IrrepFermionParity(space) => sectors_py(py, space),
            GradedSpaceInner::FermionParityU1Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::U1SU2Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::FermionParitySU2Irrep(space) => sectors_py(py, space),
            GradedSpaceInner::FermionParityU1SU2Irrep(space) => sectors_py(py, space),
        }
    }

    pub(super) fn is_dual(&self) -> bool {
        match self {
            GradedSpaceInner::Trivial(space) => space.is_dual(),
            GradedSpaceInner::U1Irrep(space) => space.is_dual(),
            GradedSpaceInner::SU2Irrep(space) => space.is_dual(),
            GradedSpaceInner::FermionParity(space) => space.is_dual(),
            GradedSpaceInner::Z2Irrep(space) => space.is_dual(),
            GradedSpaceInner::Z3Irrep(space) => space.is_dual(),
            GradedSpaceInner::Z4Irrep(space) => space.is_dual(),
            GradedSpaceInner::U1IrrepFermionParity(space) => space.is_dual(),
            GradedSpaceInner::FermionParityU1Irrep(space) => space.is_dual(),
            GradedSpaceInner::U1SU2Irrep(space) => space.is_dual(),
            GradedSpaceInner::FermionParitySU2Irrep(space) => space.is_dual(),
            GradedSpaceInner::FermionParityU1SU2Irrep(space) => space.is_dual(),
        }
    }

    pub(super) fn dual(&self) -> GradedSpaceInner {
        match self {
            GradedSpaceInner::Trivial(space) => GradedSpaceInner::Trivial(space.dual()),
            GradedSpaceInner::U1Irrep(space) => GradedSpaceInner::U1Irrep(space.dual()),
            GradedSpaceInner::SU2Irrep(space) => GradedSpaceInner::SU2Irrep(space.dual()),
            GradedSpaceInner::FermionParity(space) => GradedSpaceInner::FermionParity(space.dual()),
            GradedSpaceInner::Z2Irrep(space) => GradedSpaceInner::Z2Irrep(space.dual()),
            GradedSpaceInner::Z3Irrep(space) => GradedSpaceInner::Z3Irrep(space.dual()),
            GradedSpaceInner::Z4Irrep(space) => GradedSpaceInner::Z4Irrep(space.dual()),
            GradedSpaceInner::U1IrrepFermionParity(space) => {
                GradedSpaceInner::U1IrrepFermionParity(space.dual())
            }
            GradedSpaceInner::FermionParityU1Irrep(space) => {
                GradedSpaceInner::FermionParityU1Irrep(space.dual())
            }
            GradedSpaceInner::U1SU2Irrep(space) => GradedSpaceInner::U1SU2Irrep(space.dual()),
            GradedSpaceInner::FermionParitySU2Irrep(space) => {
                GradedSpaceInner::FermionParitySU2Irrep(space.dual())
            }
            GradedSpaceInner::FermionParityU1SU2Irrep(space) => {
                GradedSpaceInner::FermionParityU1SU2Irrep(space.dual())
            }
        }
    }

    pub(super) fn flip(&self) -> GradedSpaceInner {
        match self {
            GradedSpaceInner::Trivial(space) => GradedSpaceInner::Trivial(space.flip()),
            GradedSpaceInner::U1Irrep(space) => GradedSpaceInner::U1Irrep(space.flip()),
            GradedSpaceInner::SU2Irrep(space) => GradedSpaceInner::SU2Irrep(space.flip()),
            GradedSpaceInner::FermionParity(space) => GradedSpaceInner::FermionParity(space.flip()),
            GradedSpaceInner::Z2Irrep(space) => GradedSpaceInner::Z2Irrep(space.flip()),
            GradedSpaceInner::Z3Irrep(space) => GradedSpaceInner::Z3Irrep(space.flip()),
            GradedSpaceInner::Z4Irrep(space) => GradedSpaceInner::Z4Irrep(space.flip()),
            GradedSpaceInner::U1IrrepFermionParity(space) => {
                GradedSpaceInner::U1IrrepFermionParity(space.flip())
            }
            GradedSpaceInner::FermionParityU1Irrep(space) => {
                GradedSpaceInner::FermionParityU1Irrep(space.flip())
            }
            GradedSpaceInner::U1SU2Irrep(space) => GradedSpaceInner::U1SU2Irrep(space.flip()),
            GradedSpaceInner::FermionParitySU2Irrep(space) => {
                GradedSpaceInner::FermionParitySU2Irrep(space.flip())
            }
            GradedSpaceInner::FermionParityU1SU2Irrep(space) => {
                GradedSpaceInner::FermionParityU1SU2Irrep(space.flip())
            }
        }
    }

    fn infimum(left: &GradedSpaceInner, right: &GradedSpaceInner) -> PyResult<GradedSpaceInner> {
        match (left, right) {
            (GradedSpaceInner::Trivial(left), GradedSpaceInner::Trivial(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::Trivial)
                    .map_err(core_err)
            }
            (GradedSpaceInner::U1Irrep(left), GradedSpaceInner::U1Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::U1Irrep)
                    .map_err(core_err)
            }
            (GradedSpaceInner::SU2Irrep(left), GradedSpaceInner::SU2Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::SU2Irrep)
                    .map_err(core_err)
            }
            (GradedSpaceInner::FermionParity(left), GradedSpaceInner::FermionParity(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::FermionParity)
                    .map_err(core_err)
            }
            (GradedSpaceInner::Z2Irrep(left), GradedSpaceInner::Z2Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::Z2Irrep)
                    .map_err(core_err)
            }
            (GradedSpaceInner::Z3Irrep(left), GradedSpaceInner::Z3Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::Z3Irrep)
                    .map_err(core_err)
            }
            (GradedSpaceInner::Z4Irrep(left), GradedSpaceInner::Z4Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::Z4Irrep)
                    .map_err(core_err)
            }
            (
                GradedSpaceInner::U1IrrepFermionParity(left),
                GradedSpaceInner::U1IrrepFermionParity(right),
            ) => core_infimum_space(left, right)
                .map(GradedSpaceInner::U1IrrepFermionParity)
                .map_err(core_err),
            (
                GradedSpaceInner::FermionParityU1Irrep(left),
                GradedSpaceInner::FermionParityU1Irrep(right),
            ) => core_infimum_space(left, right)
                .map(GradedSpaceInner::FermionParityU1Irrep)
                .map_err(core_err),
            (GradedSpaceInner::U1SU2Irrep(left), GradedSpaceInner::U1SU2Irrep(right)) => {
                core_infimum_space(left, right)
                    .map(GradedSpaceInner::U1SU2Irrep)
                    .map_err(core_err)
            }
            (
                GradedSpaceInner::FermionParitySU2Irrep(left),
                GradedSpaceInner::FermionParitySU2Irrep(right),
            ) => core_infimum_space(left, right)
                .map(GradedSpaceInner::FermionParitySU2Irrep)
                .map_err(core_err),
            (
                GradedSpaceInner::FermionParityU1SU2Irrep(left),
                GradedSpaceInner::FermionParityU1SU2Irrep(right),
            ) => core_infimum_space(left, right)
                .map(GradedSpaceInner::FermionParityU1SU2Irrep)
                .map_err(core_err),
            _ => Err(PyValueError::new_err(
                "infimum spaces must have the same sector family",
            )),
        }
    }

    pub(super) fn to_spec(&self) -> ElementarySpaceSpec {
        match self {
            GradedSpaceInner::Trivial(space) => space.to_spec(),
            GradedSpaceInner::U1Irrep(space) => space.to_spec(),
            GradedSpaceInner::SU2Irrep(space) => space.to_spec(),
            GradedSpaceInner::FermionParity(space) => space.to_spec(),
            GradedSpaceInner::Z2Irrep(space) => space.to_spec(),
            GradedSpaceInner::Z3Irrep(space) => space.to_spec(),
            GradedSpaceInner::Z4Irrep(space) => space.to_spec(),
            GradedSpaceInner::U1IrrepFermionParity(space) => space.to_spec(),
            GradedSpaceInner::FermionParityU1Irrep(space) => space.to_spec(),
            GradedSpaceInner::U1SU2Irrep(space) => space.to_spec(),
            GradedSpaceInner::FermionParitySU2Irrep(space) => space.to_spec(),
            GradedSpaceInner::FermionParityU1SU2Irrep(space) => space.to_spec(),
        }
    }
}

fn build_graded_space<I: Sector>(
    dims: Vec<(Vec<i64>, usize)>,
    is_dual: bool,
) -> PyResult<GradedSpace<I>> {
    let dims = dims
        .into_iter()
        .map(|(sector, dim)| I::decode_value(&sector).map(|sector| (sector, dim)))
        .collect::<Result<Vec<_>, _>>()
        .map_err(core_err)?;
    GradedSpace::<I>::new(dims, is_dual).map_err(core_err)
}

fn build_graded_space_variant<I: Sector>(
    sectors: &Bound<'_, PyAny>,
    is_dual: bool,
    wrap: fn(GradedSpace<I>) -> GradedSpaceInner,
) -> PyResult<GradedSpaceInner> {
    let dims = parse_sector_dims(sectors, I::encoded_width())?;
    build_graded_space::<I>(dims, is_dual).map(wrap)
}

fn graded_space_from_spec(
    spec: CoreSectorSpec,
    sectors: &Bound<'_, PyAny>,
    is_dual: bool,
) -> PyResult<GradedSpaceInner> {
    match spec {
        CoreSectorSpec::Trivial => {
            build_graded_space_variant::<Trivial>(sectors, is_dual, GradedSpaceInner::Trivial)
        }
        CoreSectorSpec::Irrep {
            group: GroupSpec::U1,
        } => build_graded_space_variant::<U1Irrep>(sectors, is_dual, GradedSpaceInner::U1Irrep),
        CoreSectorSpec::Irrep {
            group: GroupSpec::SU2,
        } => build_graded_space_variant::<SU2Irrep>(sectors, is_dual, GradedSpaceInner::SU2Irrep),
        CoreSectorSpec::FermionParity => build_graded_space_variant::<FermionParity>(
            sectors,
            is_dual,
            GradedSpaceInner::FermionParity,
        ),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 2 },
        } => build_graded_space_variant::<Z2Irrep>(sectors, is_dual, GradedSpaceInner::Z2Irrep),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 3 },
        } => build_graded_space_variant::<Z3Irrep>(sectors, is_dual, GradedSpaceInner::Z3Irrep),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n: 4 },
        } => build_graded_space_variant::<Z4Irrep>(sectors, is_dual, GradedSpaceInner::Z4Irrep),
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::fermion_parity()] =>
        {
            build_graded_space_variant::<FermionNumber>(
                sectors,
                is_dual,
                GradedSpaceInner::U1IrrepFermionParity,
            )
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::u1()] =>
        {
            build_graded_space_variant::<FermionParityU1Irrep>(
                sectors,
                is_dual,
                GradedSpaceInner::FermionParityU1Irrep,
            )
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::su2()] =>
        {
            build_graded_space_variant::<U1SU2Irrep>(sectors, is_dual, GradedSpaceInner::U1SU2Irrep)
        }
        CoreSectorSpec::Product { components }
            if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::su2()] =>
        {
            build_graded_space_variant::<FermionParitySU2Irrep>(
                sectors,
                is_dual,
                GradedSpaceInner::FermionParitySU2Irrep,
            )
        }
        CoreSectorSpec::Product { components }
            if components
                == vec![
                    CoreSectorSpec::fermion_parity(),
                    CoreSectorSpec::u1(),
                    CoreSectorSpec::su2(),
                ] =>
        {
            build_graded_space_variant::<FermionParityU1SU2Irrep>(
                sectors,
                is_dual,
                GradedSpaceInner::FermionParityU1SU2Irrep,
            )
        }
        _ => Err(PyValueError::new_err("unsupported sector spec")),
    }
}
