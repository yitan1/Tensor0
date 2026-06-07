use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::layout::{
    build_degeneracy_structure as core_build_degeneracy_structure,
    build_degeneracy_structure_from_sector_structure as core_build_degeneracy_structure_from_sector_structure,
    BlockStructure, DegeneracyStructure, SubblockStructure,
};

use crate::pyconv::{core_err, tuple_from_usizes};
use crate::space::{HomSpaceInner, PyHomSpace};

use super::sector_structure::{PySectorStructure, SectorStructureInner};

#[pyclass(name = "DegeneracyStructure", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyDegeneracyStructure {
    inner: DegeneracyStructure,
}

#[pyclass(name = "BlockStructure", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyBlockStructure {
    inner: BlockStructure,
}

#[pyclass(name = "SubblockStructure", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PySubblockStructure {
    inner: SubblockStructure,
}

impl PySubblockStructure {
    pub(crate) fn from_inner(inner: SubblockStructure) -> Self {
        Self { inner }
    }
}

#[pyfunction]
pub(crate) fn build_degeneracystructure(
    space: PyRef<'_, PyHomSpace>,
) -> PyResult<PyDegeneracyStructure> {
    let inner = match space.inner() {
        HomSpaceInner::U1Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::SU2Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::FermionParity(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::Z2Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::Z3Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::Z4Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::U1IrrepFermionParity(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::FermionParityU1Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::U1SU2Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::FermionParitySU2Irrep(hom) => core_build_degeneracy_structure(hom),
        HomSpaceInner::FermionParityU1SU2Irrep(hom) => core_build_degeneracy_structure(hom),
    }
    .map_err(core_err)?;
    Ok(PyDegeneracyStructure { inner })
}

#[pyfunction]
pub(crate) fn _build_degeneracystructure_from_sectorstructure(
    space: PyRef<'_, PyHomSpace>,
    sectorstructure: PyRef<'_, PySectorStructure>,
) -> PyResult<PyDegeneracyStructure> {
    let inner = match (space.inner(), &sectorstructure.inner) {
        (HomSpaceInner::U1Irrep(hom), SectorStructureInner::U1Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (HomSpaceInner::SU2Irrep(hom), SectorStructureInner::SU2Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (HomSpaceInner::FermionParity(hom), SectorStructureInner::FermionParity(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (HomSpaceInner::Z2Irrep(hom), SectorStructureInner::Z2Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (HomSpaceInner::Z3Irrep(hom), SectorStructureInner::Z3Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (HomSpaceInner::Z4Irrep(hom), SectorStructureInner::Z4Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (
            HomSpaceInner::U1IrrepFermionParity(hom),
            SectorStructureInner::U1IrrepFermionParity(structure),
        ) => core_build_degeneracy_structure_from_sector_structure(hom, structure),
        (
            HomSpaceInner::FermionParityU1Irrep(hom),
            SectorStructureInner::FermionParityU1Irrep(structure),
        ) => core_build_degeneracy_structure_from_sector_structure(hom, structure),
        (HomSpaceInner::U1SU2Irrep(hom), SectorStructureInner::U1SU2Irrep(structure)) => {
            core_build_degeneracy_structure_from_sector_structure(hom, structure)
        }
        (
            HomSpaceInner::FermionParitySU2Irrep(hom),
            SectorStructureInner::FermionParitySU2Irrep(structure),
        ) => core_build_degeneracy_structure_from_sector_structure(hom, structure),
        (
            HomSpaceInner::FermionParityU1SU2Irrep(hom),
            SectorStructureInner::FermionParityU1SU2Irrep(structure),
        ) => core_build_degeneracy_structure_from_sector_structure(hom, structure),
        _ => {
            return Err(PyValueError::new_err(
                "sectorstructure must match HomSpace sector family",
            ));
        }
    }
    .map_err(core_err)?;
    Ok(PyDegeneracyStructure { inner })
}

#[pymethods]
impl PyDegeneracyStructure {
    #[getter]
    fn total_dim(&self) -> usize {
        self.inner.total_dim
    }

    #[getter]
    fn blockstructure(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        blockstructures_py(py, &self.inner.blockstructure)
    }

    #[getter]
    fn subblockstructure(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        subblockstructures_py(py, &self.inner.subblockstructure)
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        degeneracy_structure_static_key_py(py, &self.inner)
    }
}

#[pymethods]
impl PyBlockStructure {
    #[getter]
    fn row_dim(&self) -> usize {
        self.inner.row_dim
    }

    #[getter]
    fn col_dim(&self) -> usize {
        self.inner.col_dim
    }

    #[getter]
    fn start(&self) -> usize {
        self.inner.start
    }

    #[getter]
    fn stop(&self) -> usize {
        self.inner.stop
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        block_structure_static_key_py(py, &self.inner)
    }
}

#[pymethods]
impl PySubblockStructure {
    #[getter]
    fn sizes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        tuple_from_usizes(py, &self.inner.sizes)
    }

    #[getter]
    fn strides(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        tuple_from_usizes(py, &self.inner.strides)
    }

    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        subblock_structure_static_key_py(py, &self.inner)
    }
}

fn degeneracy_structure_static_key_py(
    py: Python<'_>,
    structure: &DegeneracyStructure,
) -> PyResult<Py<PyAny>> {
    let blocks = structure
        .blockstructure
        .iter()
        .map(|block| block_structure_static_key_py(py, block))
        .collect::<PyResult<Vec<_>>>()?;
    let subblocks = structure
        .subblockstructure
        .iter()
        .map(|subblock| subblock_structure_static_key_py(py, subblock))
        .collect::<PyResult<Vec<_>>>()?;
    let blockstructure_static_keys = PyTuple::new(py, blocks)?.into_any().unbind();
    let subblockstructure_static_keys = PyTuple::new(py, subblocks)?.into_any().unbind();
    let key = (
        "degeneracystructure",
        structure.total_dim,
        blockstructure_static_keys,
        subblockstructure_static_keys,
    )
        .into_pyobject(py)?;
    Ok(key.into_any().unbind())
}

fn blockstructures_py(py: Python<'_>, blocks: &[BlockStructure]) -> PyResult<Py<PyAny>> {
    let blocks = blocks
        .iter()
        .cloned()
        .map(|inner| Py::new(py, PyBlockStructure { inner }).map(|block| block.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, blocks)?.into_any().unbind())
}

fn subblockstructures_py(py: Python<'_>, subblocks: &[SubblockStructure]) -> PyResult<Py<PyAny>> {
    let subblocks = subblocks
        .iter()
        .cloned()
        .map(|inner| Py::new(py, PySubblockStructure { inner }).map(|subblock| subblock.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, subblocks)?.into_any().unbind())
}

fn block_structure_static_key_py(py: Python<'_>, block: &BlockStructure) -> PyResult<Py<PyAny>> {
    let key = (
        "block",
        block.row_dim,
        block.col_dim,
        block.start,
        block.stop,
    )
        .into_pyobject(py)?;
    Ok(key.into_any().unbind())
}

fn subblock_structure_static_key_py(
    py: Python<'_>,
    subblock: &SubblockStructure,
) -> PyResult<Py<PyAny>> {
    let sizes = tuple_from_usizes(py, &subblock.sizes)?;
    let strides = tuple_from_usizes(py, &subblock.strides)?;
    let key = ("subblock", sizes, strides, subblock.offset).into_pyobject(py)?;
    Ok(key.into_any().unbind())
}
