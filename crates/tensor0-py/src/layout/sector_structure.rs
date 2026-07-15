use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::fusion_tree::{FusionTree, FusionTreePair};
use tensor0_core::layout::{
    build_sector_structure as core_build_sector_structure, SectorStructure,
};
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, SU2Irrep, Sector, U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::HomSpace;

use crate::fusion_tree::{
    fusiontree_pair_py, fusiontree_pairs_py, fusiontree_static_key_py, FusionTreeInner,
    PyFusionTree,
};
use crate::pyconv::{core_err, sector_key_from_py, sectors_tuple_py};
use crate::space::{HomSpaceInner, PyHomSpace};

#[pyclass(name = "SectorStructure", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PySectorStructure {
    pub(crate) inner: SectorStructureInner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SectorStructureInner {
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

    fn blocksector_index_py(&self, sector: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        match self {
            SectorStructureInner::U1Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::SU2Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::FermionParity(structure) => {
                blocksector_index_py(structure, sector)
            }
            SectorStructureInner::Z2Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::Z3Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::Z4Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::U1IrrepFermionParity(structure) => {
                blocksector_index_py(structure, sector)
            }
            SectorStructureInner::FermionParityU1Irrep(structure) => {
                blocksector_index_py(structure, sector)
            }
            SectorStructureInner::U1SU2Irrep(structure) => blocksector_index_py(structure, sector),
            SectorStructureInner::FermionParitySU2Irrep(structure) => {
                blocksector_index_py(structure, sector)
            }
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => {
                blocksector_index_py(structure, sector)
            }
        }
    }

    fn fusiontree_pair_count(&self) -> usize {
        match self {
            SectorStructureInner::U1Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::SU2Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::FermionParity(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::Z2Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::Z3Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::Z4Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::U1IrrepFermionParity(structure) => {
                structure.fusiontree_pair_count()
            }
            SectorStructureInner::FermionParityU1Irrep(structure) => {
                structure.fusiontree_pair_count()
            }
            SectorStructureInner::U1SU2Irrep(structure) => structure.fusiontree_pair_count(),
            SectorStructureInner::FermionParitySU2Irrep(structure) => {
                structure.fusiontree_pair_count()
            }
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => {
                structure.fusiontree_pair_count()
            }
        }
    }

    fn fusiontree_pair_at_py(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        match self {
            SectorStructureInner::U1Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::U1Irrep)
            }
            SectorStructureInner::SU2Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::SU2Irrep)
            }
            SectorStructureInner::FermionParity(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::FermionParity)
            }
            SectorStructureInner::Z2Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::Z2Irrep)
            }
            SectorStructureInner::Z3Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::Z3Irrep)
            }
            SectorStructureInner::Z4Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::Z4Irrep)
            }
            SectorStructureInner::U1IrrepFermionParity(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::U1IrrepFermionParity)
            }
            SectorStructureInner::FermionParityU1Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::FermionParityU1Irrep)
            }
            SectorStructureInner::U1SU2Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::U1SU2Irrep)
            }
            SectorStructureInner::FermionParitySU2Irrep(structure) => {
                fusiontree_pair_at_py(py, structure, index, FusionTreeInner::FermionParitySU2Irrep)
            }
            SectorStructureInner::FermionParityU1SU2Irrep(structure) => fusiontree_pair_at_py(
                py,
                structure,
                index,
                FusionTreeInner::FermionParityU1SU2Irrep,
            ),
        }
    }

    fn fusiontree_pair_index(&self, row: &FusionTreeInner, col: &FusionTreeInner) -> Option<usize> {
        match (self, row, col) {
            (
                SectorStructureInner::U1Irrep(structure),
                FusionTreeInner::U1Irrep(row),
                FusionTreeInner::U1Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::SU2Irrep(structure),
                FusionTreeInner::SU2Irrep(row),
                FusionTreeInner::SU2Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::FermionParity(structure),
                FusionTreeInner::FermionParity(row),
                FusionTreeInner::FermionParity(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::Z2Irrep(structure),
                FusionTreeInner::Z2Irrep(row),
                FusionTreeInner::Z2Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::Z3Irrep(structure),
                FusionTreeInner::Z3Irrep(row),
                FusionTreeInner::Z3Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::Z4Irrep(structure),
                FusionTreeInner::Z4Irrep(row),
                FusionTreeInner::Z4Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::U1IrrepFermionParity(structure),
                FusionTreeInner::U1IrrepFermionParity(row),
                FusionTreeInner::U1IrrepFermionParity(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::FermionParityU1Irrep(structure),
                FusionTreeInner::FermionParityU1Irrep(row),
                FusionTreeInner::FermionParityU1Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::U1SU2Irrep(structure),
                FusionTreeInner::U1SU2Irrep(row),
                FusionTreeInner::U1SU2Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::FermionParitySU2Irrep(structure),
                FusionTreeInner::FermionParitySU2Irrep(row),
                FusionTreeInner::FermionParitySU2Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            (
                SectorStructureInner::FermionParityU1SU2Irrep(structure),
                FusionTreeInner::FermionParityU1SU2Irrep(row),
                FusionTreeInner::FermionParityU1SU2Irrep(col),
            ) => fusiontree_pair_index(structure, row, col),
            _ => None,
        }
    }
}

#[pyfunction]
pub(crate) fn build_sectorstructure(space: PyRef<'_, PyHomSpace>) -> PyResult<PySectorStructure> {
    let inner = SectorStructureInner::from_hom(space.inner())?;
    Ok(PySectorStructure { inner })
}

#[pyfunction]
pub(crate) fn unique_fusiontree_pair_index(
    space: PyRef<'_, PyHomSpace>,
    sectorstructure: PyRef<'_, PySectorStructure>,
    visible_sectors: &Bound<'_, PyTuple>,
) -> PyResult<Option<usize>> {
    match (space.inner(), &sectorstructure.inner) {
        (HomSpaceInner::U1Irrep(hom), SectorStructureInner::U1Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (HomSpaceInner::SU2Irrep(hom), SectorStructureInner::SU2Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (HomSpaceInner::FermionParity(hom), SectorStructureInner::FermionParity(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (HomSpaceInner::Z2Irrep(hom), SectorStructureInner::Z2Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (HomSpaceInner::Z3Irrep(hom), SectorStructureInner::Z3Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (HomSpaceInner::Z4Irrep(hom), SectorStructureInner::Z4Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (
            HomSpaceInner::U1IrrepFermionParity(hom),
            SectorStructureInner::U1IrrepFermionParity(structure),
        ) => unique_fusiontree_pair_index_py(hom, structure, visible_sectors),
        (
            HomSpaceInner::FermionParityU1Irrep(hom),
            SectorStructureInner::FermionParityU1Irrep(structure),
        ) => unique_fusiontree_pair_index_py(hom, structure, visible_sectors),
        (HomSpaceInner::U1SU2Irrep(hom), SectorStructureInner::U1SU2Irrep(structure)) => {
            unique_fusiontree_pair_index_py(hom, structure, visible_sectors)
        }
        (
            HomSpaceInner::FermionParitySU2Irrep(hom),
            SectorStructureInner::FermionParitySU2Irrep(structure),
        ) => unique_fusiontree_pair_index_py(hom, structure, visible_sectors),
        (
            HomSpaceInner::FermionParityU1SU2Irrep(hom),
            SectorStructureInner::FermionParityU1SU2Irrep(structure),
        ) => unique_fusiontree_pair_index_py(hom, structure, visible_sectors),
        _ => Err(PyValueError::new_err(
            "sectorstructure must match HomSpace sector family",
        )),
    }
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
    fn fusiontree_pair_count(&self) -> usize {
        self.inner.fusiontree_pair_count()
    }

    fn fusiontree_pair_at(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyAny>>> {
        self.inner.fusiontree_pair_at_py(py, index)
    }

    fn fusiontree_pair_index(
        &self,
        row: PyRef<'_, PyFusionTree>,
        col: PyRef<'_, PyFusionTree>,
    ) -> Option<usize> {
        self.inner.fusiontree_pair_index(&row.inner, &col.inner)
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.static_key_py(py)
    }

    fn blocksector_index(&self, sector: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        self.inner.blocksector_index_py(sector)
    }
}

fn fusiontree_pair_at_py<I, F>(
    py: Python<'_>,
    structure: &SectorStructure<I>,
    index: usize,
    wrap: F,
) -> PyResult<Option<Py<PyAny>>>
where
    I: Sector,
    F: Fn(FusionTree<I>) -> FusionTreeInner + Copy,
{
    structure
        .fusiontree_pair_at(index)
        .map(|pair| fusiontree_pair_py(py, pair, wrap))
        .transpose()
}

fn fusiontree_pair_index<I: Sector>(
    structure: &SectorStructure<I>,
    row: &FusionTree<I>,
    col: &FusionTree<I>,
) -> Option<usize> {
    structure.fusiontree_pair_index(&FusionTreePair {
        row: row.clone(),
        col: col.clone(),
    })
}

fn unique_fusiontree_pair_index_py<I: Sector>(
    space: &HomSpace<I>,
    structure: &SectorStructure<I>,
    visible_sectors: &Bound<'_, PyTuple>,
) -> PyResult<Option<usize>> {
    if visible_sectors.len() != space.numind() {
        return Err(PyValueError::new_err(format!(
            "TensorMap sector indices must have length {}",
            space.numind(),
        )));
    }

    let visible_sectors = visible_sectors
        .iter()
        .map(|sector| {
            let encoded = sector_key_from_py(&sector)?;
            I::decode_value(&encoded).map_err(core_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let row_uncoupled = &visible_sectors[..space.numout()];
    let col_uncoupled = visible_sectors[space.numout()..]
        .iter()
        .map(Sector::dual)
        .collect::<Vec<_>>();

    structure
        .unique_fusiontree_pair_index(space, row_uncoupled, &col_uncoupled)
        .map_err(core_err)
}

fn blocksector_index_py<I: Sector>(
    structure: &SectorStructure<I>,
    sector: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
    let encoded = sector_key_from_py(sector)?;
    let typed_sector = I::decode_value(&encoded).map_err(core_err)?;
    Ok(structure.blocksector_index(&typed_sector))
}

fn sector_structure_static_key_py<I: Sector>(
    py: Python<'_>,
    structure: &SectorStructure<I>,
) -> PyResult<Py<PyAny>> {
    let blocksectors = sectors_tuple_py(py, structure.blocksectors())?;
    let pairs = structure
        .fusiontree_pairs()
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
