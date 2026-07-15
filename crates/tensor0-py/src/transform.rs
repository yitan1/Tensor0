use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAny, PyTuple};
use tensor0_core::transform::{
    flip_entries as core_flip_entries,
    reweighting::{
        twist_is_trivial as core_twist_is_trivial,
        twist_subblock_factors as core_twist_subblock_factors,
    },
    trace_transformer as core_trace_transformer, tree_braider as core_tree_braider,
    tree_transposer as core_tree_transposer, AbelianTransformData, GenericTransformData,
    TreeTransformer,
};

use crate::layout::{PySectorStructure, SectorStructureInner};
use crate::pyconv::{core_err, tuple_from_usizes};
use crate::space::{HomSpaceInner, PyHomSpace};

#[pyclass(name = "TreeTransformer", skip_from_py_object)]
pub(crate) struct PyTreeTransformer {
    inner: TreeTransformer,
    abelian_data: PyOnceLock<Py<PyAny>>,
    generic_data: PyOnceLock<Py<PyAny>>,
}

#[pyclass(name = "AbelianTransformData", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyAbelianTransformData {
    inner: AbelianTransformData,
}

#[pyclass(name = "GenericTransformData", skip_from_py_object)]
pub(crate) struct PyGenericTransformData {
    src_indices: Py<PyAny>,
    dst_indices: Py<PyAny>,
    transform: Py<PyArray2<f64>>,
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

macro_rules! dispatch_matching_homspaces_and_structures {
    ($src:expr, $dst:expr, $src_structure:expr, $dst_structure:expr, |$src_hom:ident, $dst_hom:ident, $src_layout:ident, $dst_layout:ident| $body:block) => {
        match (
            $src.inner(),
            $dst.inner(),
            &$src_structure.inner,
            &$dst_structure.inner,
        ) {
            (
                HomSpaceInner::U1Irrep($src_hom),
                HomSpaceInner::U1Irrep($dst_hom),
                SectorStructureInner::U1Irrep($src_layout),
                SectorStructureInner::U1Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::SU2Irrep($src_hom),
                HomSpaceInner::SU2Irrep($dst_hom),
                SectorStructureInner::SU2Irrep($src_layout),
                SectorStructureInner::SU2Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::FermionParity($src_hom),
                HomSpaceInner::FermionParity($dst_hom),
                SectorStructureInner::FermionParity($src_layout),
                SectorStructureInner::FermionParity($dst_layout),
            ) => $body,
            (
                HomSpaceInner::Z2Irrep($src_hom),
                HomSpaceInner::Z2Irrep($dst_hom),
                SectorStructureInner::Z2Irrep($src_layout),
                SectorStructureInner::Z2Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::Z3Irrep($src_hom),
                HomSpaceInner::Z3Irrep($dst_hom),
                SectorStructureInner::Z3Irrep($src_layout),
                SectorStructureInner::Z3Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::Z4Irrep($src_hom),
                HomSpaceInner::Z4Irrep($dst_hom),
                SectorStructureInner::Z4Irrep($src_layout),
                SectorStructureInner::Z4Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::U1IrrepFermionParity($src_hom),
                HomSpaceInner::U1IrrepFermionParity($dst_hom),
                SectorStructureInner::U1IrrepFermionParity($src_layout),
                SectorStructureInner::U1IrrepFermionParity($dst_layout),
            ) => $body,
            (
                HomSpaceInner::FermionParityU1Irrep($src_hom),
                HomSpaceInner::FermionParityU1Irrep($dst_hom),
                SectorStructureInner::FermionParityU1Irrep($src_layout),
                SectorStructureInner::FermionParityU1Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::U1SU2Irrep($src_hom),
                HomSpaceInner::U1SU2Irrep($dst_hom),
                SectorStructureInner::U1SU2Irrep($src_layout),
                SectorStructureInner::U1SU2Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::FermionParitySU2Irrep($src_hom),
                HomSpaceInner::FermionParitySU2Irrep($dst_hom),
                SectorStructureInner::FermionParitySU2Irrep($src_layout),
                SectorStructureInner::FermionParitySU2Irrep($dst_layout),
            ) => $body,
            (
                HomSpaceInner::FermionParityU1SU2Irrep($src_hom),
                HomSpaceInner::FermionParityU1SU2Irrep($dst_hom),
                SectorStructureInner::FermionParityU1SU2Irrep($src_layout),
                SectorStructureInner::FermionParityU1SU2Irrep($dst_layout),
            ) => $body,
            _ => Err(PyValueError::new_err(
                "source, destination, and sectorstructures must use the same sector family",
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
pub(crate) fn trace_transformer(
    canonical_src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    canonical_sectorstructure: PyRef<'_, PySectorStructure>,
    dst_sectorstructure: PyRef<'_, PySectorStructure>,
    basis_transformer: PyRef<'_, PyTreeTransformer>,
) -> PyResult<PyTreeTransformer> {
    let inner = dispatch_matching_homspaces_and_structures!(
        canonical_src,
        dst,
        canonical_sectorstructure,
        dst_sectorstructure,
        |canonical_hom, dst_hom, canonical_structure, dst_structure| {
            core_trace_transformer(
                canonical_hom,
                dst_hom,
                canonical_structure,
                dst_structure,
                &basis_transformer.inner,
            )
            .map_err(core_err)
        }
    )?;
    Ok(PyTreeTransformer::from_inner(inner))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tree_braider(
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    src_sectorstructure: PyRef<'_, PySectorStructure>,
    dst_sectorstructure: PyRef<'_, PySectorStructure>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
    levels_codomain: Vec<usize>,
    levels_domain: Vec<usize>,
) -> PyResult<PyTreeTransformer> {
    let inner = dispatch_matching_homspaces_and_structures!(
        src,
        dst,
        src_sectorstructure,
        dst_sectorstructure,
        |src_hom, dst_hom, src_structure, dst_structure| {
            core_tree_braider(
                src_hom,
                dst_hom,
                src_structure,
                dst_structure,
                &p_codomain,
                &p_domain,
                &levels_codomain,
                &levels_domain,
            )
            .map_err(core_err)
        }
    )?;

    Ok(PyTreeTransformer::from_inner(inner))
}

#[pyfunction]
pub(crate) fn tree_transposer(
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    src_sectorstructure: PyRef<'_, PySectorStructure>,
    dst_sectorstructure: PyRef<'_, PySectorStructure>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
) -> PyResult<PyTreeTransformer> {
    let inner = dispatch_matching_homspaces_and_structures!(
        src,
        dst,
        src_sectorstructure,
        dst_sectorstructure,
        |src_hom, dst_hom, src_structure, dst_structure| {
            core_tree_transposer(
                src_hom,
                dst_hom,
                src_structure,
                dst_structure,
                &p_codomain,
                &p_domain,
            )
            .map_err(core_err)
        }
    )?;

    Ok(PyTreeTransformer::from_inner(inner))
}

#[pyfunction(signature = (src, dst, src_sectorstructure, dst_sectorstructure, indices, inv=false))]
pub(crate) fn flip_entries(
    py: Python<'_>,
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    src_sectorstructure: PyRef<'_, PySectorStructure>,
    dst_sectorstructure: PyRef<'_, PySectorStructure>,
    indices: Vec<usize>,
    inv: bool,
) -> PyResult<Py<PyAny>> {
    let entries = dispatch_matching_homspaces_and_structures!(
        src,
        dst,
        src_sectorstructure,
        dst_sectorstructure,
        |src_hom, dst_hom, src_structure, dst_structure| {
            core_flip_entries(
                src_hom,
                dst_hom,
                src_structure,
                dst_structure,
                &indices,
                inv,
            )
            .map_err(core_err)
        }
    )?;
    Ok(PyTuple::new(py, entries)?.into_any().unbind())
}

impl PyTreeTransformer {
    fn from_inner(inner: TreeTransformer) -> Self {
        Self {
            inner,
            abelian_data: PyOnceLock::new(),
            generic_data: PyOnceLock::new(),
        }
    }

    fn build_abelian_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

    fn build_generic_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let TreeTransformer::Generic(data) = &self.inner else {
            return Ok(PyTuple::empty(py).into_any().unbind());
        };
        let entries = data
            .iter()
            .map(|inner| {
                PyGenericTransformData::from_inner(py, inner)
                    .and_then(|entry| Py::new(py, entry))
                    .map(|entry| entry.into_any())
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries)?.into_any().unbind())
    }
}

impl PyGenericTransformData {
    fn from_inner(py: Python<'_>, inner: &GenericTransformData) -> PyResult<Self> {
        let transform = inner.transform.clone().into_pyarray(py);
        transform.call_method1("setflags", (false,))?;
        Ok(Self {
            src_indices: tuple_from_usizes(py, &inner.src_indices)?,
            dst_indices: tuple_from_usizes(py, &inner.dst_indices)?,
            transform: transform.unbind(),
        })
    }
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
    fn has_only_unit_coefficients(&self) -> bool {
        match &self.inner {
            TreeTransformer::Abelian(data) => data
                .iter()
                .all(|entry| entry.coeff == 1.0 || entry.coeff == -1.0),
            TreeTransformer::Generic(data) => data.iter().all(|group| {
                group
                    .transform
                    .iter()
                    .all(|value| *value == -1.0 || *value == 0.0 || *value == 1.0)
            }),
        }
    }

    #[getter]
    fn abelian_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.abelian_data
            .get_or_try_init(py, || self.build_abelian_data(py))
            .map(|data| data.clone_ref(py))
    }

    #[getter]
    fn generic_data(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.generic_data
            .get_or_try_init(py, || self.build_generic_data(py))
            .map(|data| data.clone_ref(py))
    }
}

#[pymethods]
impl PyAbelianTransformData {
    #[getter]
    fn coeff(&self) -> f64 {
        self.inner.coeff
    }

    #[getter]
    fn src(&self) -> usize {
        self.inner.src
    }

    #[getter]
    fn dst(&self) -> usize {
        self.inner.dst
    }
}

#[pymethods]
impl PyGenericTransformData {
    #[getter]
    fn src_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.src_indices.clone_ref(py)
    }

    #[getter]
    fn dst_indices(&self, py: Python<'_>) -> Py<PyAny> {
        self.dst_indices.clone_ref(py)
    }

    #[getter]
    fn transform(&self, py: Python<'_>) -> Py<PyArray2<f64>> {
        self.transform.clone_ref(py)
    }
}
