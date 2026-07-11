use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::transform::{
    reweighting::{
        twist_is_trivial as core_twist_is_trivial,
        twist_subblock_factors as core_twist_subblock_factors,
    },
    tree_braider as core_tree_braider, tree_transposer as core_tree_transposer,
    AbelianTransformData, GenericTransformData, GenericTransformStructures, TreeTransformer,
};

use crate::layout::{PySectorStructure, PySubblockStructure, SectorStructureInner};
use crate::pyconv::{core_err, tuple_from_usizes};
use crate::space::{HomSpaceInner, PyHomSpace};

#[pyclass(name = "TreeTransformer", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyTreeTransformer {
    inner: TreeTransformer,
}

#[pyclass(name = "AbelianTransformData", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyAbelianTransformData {
    inner: AbelianTransformData,
}

#[pyclass(name = "GenericTransformData", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyGenericTransformData {
    inner: GenericTransformData,
}

#[pyclass(name = "GenericTransformStructures", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyGenericTransformStructures {
    inner: GenericTransformStructures,
}

macro_rules! dispatch_matching_homspaces {
    ($src:expr, $dst:expr, |$src_hom:ident, $dst_hom:ident| $body:block) => {
        match ($src.inner(), $dst.inner()) {
            (HomSpaceInner::U1Irrep($src_hom), HomSpaceInner::U1Irrep($dst_hom)) => $body,
            (HomSpaceInner::SU2Irrep($src_hom), HomSpaceInner::SU2Irrep($dst_hom)) => $body,
            (HomSpaceInner::FermionParity($src_hom), HomSpaceInner::FermionParity($dst_hom)) => {
                $body
            }
            (HomSpaceInner::Z2Irrep($src_hom), HomSpaceInner::Z2Irrep($dst_hom)) => $body,
            (HomSpaceInner::Z3Irrep($src_hom), HomSpaceInner::Z3Irrep($dst_hom)) => $body,
            (HomSpaceInner::Z4Irrep($src_hom), HomSpaceInner::Z4Irrep($dst_hom)) => $body,
            (
                HomSpaceInner::U1IrrepFermionParity($src_hom),
                HomSpaceInner::U1IrrepFermionParity($dst_hom),
            ) => $body,
            (
                HomSpaceInner::FermionParityU1Irrep($src_hom),
                HomSpaceInner::FermionParityU1Irrep($dst_hom),
            ) => $body,
            (HomSpaceInner::U1SU2Irrep($src_hom), HomSpaceInner::U1SU2Irrep($dst_hom)) => $body,
            (
                HomSpaceInner::FermionParitySU2Irrep($src_hom),
                HomSpaceInner::FermionParitySU2Irrep($dst_hom),
            ) => $body,
            (
                HomSpaceInner::FermionParityU1SU2Irrep($src_hom),
                HomSpaceInner::FermionParityU1SU2Irrep($dst_hom),
            ) => $body,
            _ => Err(PyValueError::new_err(
                "src and dst HomSpace sector families must match",
            )),
        }
    };
}

macro_rules! dispatch_twist_is_trivial {
    ($space:expr, $indices:expr) => {
        match $space.inner() {
            HomSpaceInner::U1Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::SU2Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::FermionParity(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::Z2Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::Z3Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::Z4Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::U1IrrepFermionParity(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::FermionParityU1Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::U1SU2Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::FermionParitySU2Irrep(hom) => core_twist_is_trivial(hom, $indices),
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => core_twist_is_trivial(hom, $indices),
        }
    };
}

macro_rules! dispatch_twist_subblock_factors {
    ($space:expr, $structure:expr, $indices:expr, $inv:expr) => {
        match ($space.inner(), &$structure.inner) {
            (HomSpaceInner::U1Irrep(hom), SectorStructureInner::U1Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (HomSpaceInner::SU2Irrep(hom), SectorStructureInner::SU2Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (HomSpaceInner::FermionParity(hom), SectorStructureInner::FermionParity(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (HomSpaceInner::Z2Irrep(hom), SectorStructureInner::Z2Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (HomSpaceInner::Z3Irrep(hom), SectorStructureInner::Z3Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (HomSpaceInner::Z4Irrep(hom), SectorStructureInner::Z4Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (
                HomSpaceInner::U1IrrepFermionParity(hom),
                SectorStructureInner::U1IrrepFermionParity(structure),
            ) => core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err),
            (
                HomSpaceInner::FermionParityU1Irrep(hom),
                SectorStructureInner::FermionParityU1Irrep(structure),
            ) => core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err),
            (HomSpaceInner::U1SU2Irrep(hom), SectorStructureInner::U1SU2Irrep(structure)) => {
                core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err)
            }
            (
                HomSpaceInner::FermionParitySU2Irrep(hom),
                SectorStructureInner::FermionParitySU2Irrep(structure),
            ) => core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err),
            (
                HomSpaceInner::FermionParityU1SU2Irrep(hom),
                SectorStructureInner::FermionParityU1SU2Irrep(structure),
            ) => core_twist_subblock_factors(hom, structure, $indices, $inv).map_err(core_err),
            _ => Err(PyValueError::new_err(
                "sectorstructure must match HomSpace sector family",
            )),
        }
    };
}

#[pyfunction]
pub(crate) fn twist_is_trivial(
    space: PyRef<'_, PyHomSpace>,
    indices: Vec<usize>,
) -> PyResult<bool> {
    dispatch_twist_is_trivial!(space, &indices).map_err(core_err)
}

#[pyfunction(signature = (space, sectorstructure, indices, inv=false))]
pub(crate) fn twist_subblock_factors(
    py: Python<'_>,
    space: PyRef<'_, PyHomSpace>,
    sectorstructure: PyRef<'_, PySectorStructure>,
    indices: Vec<usize>,
    inv: bool,
) -> PyResult<Option<Py<PyAny>>> {
    let factors = dispatch_twist_subblock_factors!(space, sectorstructure, &indices, inv)?;
    match factors {
        Some(factors) => Ok(Some(PyTuple::new(py, factors)?.into_any().unbind())),
        None => Ok(None),
    }
}

#[pyfunction]
pub(crate) fn tree_braider(
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
    levels_codomain: Vec<usize>,
    levels_domain: Vec<usize>,
) -> PyResult<PyTreeTransformer> {
    let inner = dispatch_matching_homspaces!(src, dst, |src_hom, dst_hom| {
        core_tree_braider(
            src_hom,
            dst_hom,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain,
        )
        .map_err(core_err)
    })?;

    Ok(PyTreeTransformer { inner })
}

#[pyfunction]
pub(crate) fn tree_transposer(
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
) -> PyResult<PyTreeTransformer> {
    let inner = dispatch_matching_homspaces!(src, dst, |src_hom, dst_hom| {
        core_tree_transposer(src_hom, dst_hom, &p_codomain, &p_domain).map_err(core_err)
    })?;

    Ok(PyTreeTransformer { inner })
}

#[pymethods]
impl PyTreeTransformer {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            TreeTransformer::Abelian(_) => "abelian",
            TreeTransformer::Generic(_) => "generic",
        }
    }

    #[getter]
    fn abelian_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let TreeTransformer::Abelian(data) = &self.inner else {
            return Ok(PyTuple::empty(py).into_any().unbind());
        };
        let entries = data
            .iter()
            .cloned()
            .map(|inner| {
                Py::new(py, PyAbelianTransformData { inner }).map(|entry| entry.into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.into_any().unbind())
    }

    #[getter]
    fn generic_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let TreeTransformer::Generic(data) = &self.inner else {
            return Ok(PyTuple::empty(py).into_any().unbind());
        };
        let entries = data
            .iter()
            .cloned()
            .map(|inner| {
                Py::new(py, PyGenericTransformData { inner }).map(|entry| entry.into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.into_any().unbind())
    }
}

#[pymethods]
impl PyAbelianTransformData {
    #[getter]
    fn coeff(&self) -> f64 {
        self.inner.coeff
    }

    #[getter]
    fn src(&self, py: Python<'_>) -> PyResult<Py<PySubblockStructure>> {
        Py::new(py, PySubblockStructure::from_inner(self.inner.src.clone()))
    }

    #[getter]
    fn dst(&self, py: Python<'_>) -> PyResult<Py<PySubblockStructure>> {
        Py::new(py, PySubblockStructure::from_inner(self.inner.dst.clone()))
    }
}

#[pymethods]
impl PyGenericTransformData {
    #[getter]
    fn src(&self, py: Python<'_>) -> PyResult<Py<PyGenericTransformStructures>> {
        Py::new(
            py,
            PyGenericTransformStructures {
                inner: self.inner.src.clone(),
            },
        )
    }

    #[getter]
    fn dst(&self, py: Python<'_>) -> PyResult<Py<PyGenericTransformStructures>> {
        Py::new(
            py,
            PyGenericTransformStructures {
                inner: self.inner.dst.clone(),
            },
        )
    }

    #[getter]
    fn basis_transform(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.inner.basis_transform.clone().into_pyarray(py).unbind()
    }
}

#[pymethods]
impl PyGenericTransformStructures {
    #[getter]
    fn sizes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        tuple_from_usizes(py, &self.inner.sizes)
    }

    #[getter]
    fn strides_offsets(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let entries = self
            .inner
            .strides_offsets
            .iter()
            .map(|(strides, offset)| {
                let strides = tuple_from_usizes(py, strides)?;
                let entry = (strides, *offset).into_pyobject(py)?;
                Ok(entry.into_any().unbind())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.into_any().unbind())
    }
}
