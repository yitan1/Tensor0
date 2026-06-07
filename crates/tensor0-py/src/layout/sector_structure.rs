use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::layout::{
    build_sector_structure as core_build_sector_structure, SectorStructure,
};
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, SU2Irrep, Sector, U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};

use crate::fusion_tree::{fusiontree_pairs_py, fusiontree_static_key_py, FusionTreeInner};
use crate::pyconv::{core_err, sectors_tuple_py};
use crate::space::{HomSpaceInner, PyHomSpace};

#[pyclass(name = "SectorStructure", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PySectorStructure {
    pub(in crate::layout) inner: SectorStructureInner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::layout) enum SectorStructureInner {
    U1Irrep(SectorStructure<U1Irrep>),
    SU2Irrep(SectorStructure<SU2Irrep>),
    FermionParity(SectorStructure<FermionParity>),
    Z2Irrep(SectorStructure<Z2Irrep>),
    Z3Irrep(SectorStructure<Z3Irrep>),
    Z4Irrep(SectorStructure<Z4Irrep>),
    U1IrrepFermionParity(SectorStructure<FermionNumber>),
    FermionParityU1Irrep(SectorStructure<FermionParityU1Irrep>),
    U1SU2Irrep(SectorStructure<U1SU2Irrep>),
    FermionParitySU2Irrep(SectorStructure<FermionParitySU2Irrep>),
    FermionParityU1SU2Irrep(SectorStructure<FermionParityU1SU2Irrep>),
}

impl SectorStructureInner {
    fn from_hom(space: &HomSpaceInner) -> PyResult<Self> {
        match space {
            HomSpaceInner::U1Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::U1Irrep)
                .map_err(core_err),
            HomSpaceInner::SU2Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::SU2Irrep)
                .map_err(core_err),
            HomSpaceInner::FermionParity(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::FermionParity)
                .map_err(core_err),
            HomSpaceInner::Z2Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::Z2Irrep)
                .map_err(core_err),
            HomSpaceInner::Z3Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::Z3Irrep)
                .map_err(core_err),
            HomSpaceInner::Z4Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::Z4Irrep)
                .map_err(core_err),
            HomSpaceInner::U1IrrepFermionParity(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::U1IrrepFermionParity)
                .map_err(core_err),
            HomSpaceInner::FermionParityU1Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::FermionParityU1Irrep)
                .map_err(core_err),
            HomSpaceInner::U1SU2Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::U1SU2Irrep)
                .map_err(core_err),
            HomSpaceInner::FermionParitySU2Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::FermionParitySU2Irrep)
                .map_err(core_err),
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => core_build_sector_structure(hom)
                .map(SectorStructureInner::FermionParityU1SU2Irrep)
                .map_err(core_err),
        }
    }

    fn blocksectors_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            SectorStructureInner::U1Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::SU2Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::FermionParity(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::Z2Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::Z3Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::Z4Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::U1IrrepFermionParity(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::FermionParityU1Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::U1SU2Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::FermionParitySU2Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => {
                sectors_tuple_py(py, structure.blocksectors())
            }
        }
    }

    fn fusiontree_pairs_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            SectorStructureInner::U1Irrep(structure) => {
                fusiontree_pairs_py(py, structure.fusiontree_pairs(), FusionTreeInner::U1Irrep)
            }
            SectorStructureInner::SU2Irrep(structure) => {
                fusiontree_pairs_py(py, structure.fusiontree_pairs(), FusionTreeInner::SU2Irrep)
            }
            SectorStructureInner::FermionParity(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::FermionParity,
            ),
            SectorStructureInner::Z2Irrep(structure) => {
                fusiontree_pairs_py(py, structure.fusiontree_pairs(), FusionTreeInner::Z2Irrep)
            }
            SectorStructureInner::Z3Irrep(structure) => {
                fusiontree_pairs_py(py, structure.fusiontree_pairs(), FusionTreeInner::Z3Irrep)
            }
            SectorStructureInner::Z4Irrep(structure) => {
                fusiontree_pairs_py(py, structure.fusiontree_pairs(), FusionTreeInner::Z4Irrep)
            }
            SectorStructureInner::U1IrrepFermionParity(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::U1IrrepFermionParity,
            ),
            SectorStructureInner::FermionParityU1Irrep(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::FermionParityU1Irrep,
            ),
            SectorStructureInner::U1SU2Irrep(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::U1SU2Irrep,
            ),
            SectorStructureInner::FermionParitySU2Irrep(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::FermionParitySU2Irrep,
            ),
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => fusiontree_pairs_py(
                py,
                structure.fusiontree_pairs(),
                FusionTreeInner::FermionParityU1SU2Irrep,
            ),
        }
    }

    fn static_key_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            SectorStructureInner::U1Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::SU2Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::FermionParity(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::Z2Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::Z3Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::Z4Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::U1IrrepFermionParity(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::FermionParityU1Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::U1SU2Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::FermionParitySU2Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => {
                sector_structure_static_key_py(py, structure)
            }
        }
    }
}

#[pyfunction]
pub(crate) fn build_sectorstructure(space: PyRef<'_, PyHomSpace>) -> PyResult<PySectorStructure> {
    let inner = SectorStructureInner::from_hom(space.inner())?;
    Ok(PySectorStructure { inner })
}

#[pymethods]
impl PySectorStructure {
    #[getter]
    fn blocksectors(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.blocksectors_py(py)
    }

    #[getter]
    fn fusiontree_pairs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.fusiontree_pairs_py(py)
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.static_key_py(py)
    }
}

fn sector_structure_static_key_py<I: Sector>(
    py: Python<'_>,
    structure: &SectorStructure<I>,
) -> PyResult<Py<PyAny>> {
    let blocksectors = sectors_tuple_py(py, structure.blocksectors())?;
    let pairs = structure
        .fusiontree_pairs()
        .iter()
        .map(|pair| {
            let row = fusiontree_static_key_py(py, &pair.row)?;
            let col = fusiontree_static_key_py(py, &pair.col)?;
            let pair = (row, col).into_pyobject(py)?;
            Ok(pair.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let fusiontree_pair_static_keys = PyTuple::new(py, pairs)?.into_any().unbind();
    let key = ("sectorstructure", blocksectors, fusiontree_pair_static_keys).into_pyobject(py)?;
    Ok(key.into_any().unbind())
}
