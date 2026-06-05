use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::transform::{
    tree_braider as core_tree_braider, tree_transposer as core_tree_transposer,
    AbelianTransformData, GenericTransformData, GenericTransformStructures, TreeTransformer,
};

use crate::layout::PySubblockStructure;
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

macro_rules! tree_braider_hom {
    ($src:expr, $dst:expr, $p_codomain:expr, $p_domain:expr, $levels_codomain:expr, $levels_domain:expr) => {
        core_tree_braider(
            $src,
            $dst,
            $p_codomain,
            $p_domain,
            $levels_codomain,
            $levels_domain,
        )
        .map_err(core_err)
    };
}

macro_rules! tree_transposer_hom {
    ($src:expr, $dst:expr, $p_codomain:expr, $p_domain:expr) => {
        core_tree_transposer($src, $dst, $p_codomain, $p_domain).map_err(core_err)
    };
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
    let inner = match (src.inner(), dst.inner()) {
        (HomSpaceInner::U1Irrep(src), HomSpaceInner::U1Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::SU2Irrep(src), HomSpaceInner::SU2Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::FermionParity(src), HomSpaceInner::FermionParity(dst)) => {
            tree_braider_hom!(
                src,
                dst,
                &p_codomain,
                &p_domain,
                &levels_codomain,
                &levels_domain
            )
        }
        (HomSpaceInner::Z2Irrep(src), HomSpaceInner::Z2Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::Z3Irrep(src), HomSpaceInner::Z3Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::Z4Irrep(src), HomSpaceInner::Z4Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::U1IrrepFermionParity(src), HomSpaceInner::U1IrrepFermionParity(dst)) => {
            tree_braider_hom!(
                src,
                dst,
                &p_codomain,
                &p_domain,
                &levels_codomain,
                &levels_domain
            )
        }
        (HomSpaceInner::FermionParityU1Irrep(src), HomSpaceInner::FermionParityU1Irrep(dst)) => {
            tree_braider_hom!(
                src,
                dst,
                &p_codomain,
                &p_domain,
                &levels_codomain,
                &levels_domain
            )
        }
        (HomSpaceInner::U1SU2Irrep(src), HomSpaceInner::U1SU2Irrep(dst)) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        (HomSpaceInner::FermionParitySU2Irrep(src), HomSpaceInner::FermionParitySU2Irrep(dst)) => {
            tree_braider_hom!(
                src,
                dst,
                &p_codomain,
                &p_domain,
                &levels_codomain,
                &levels_domain
            )
        }
        (
            HomSpaceInner::FermionParityU1SU2Irrep(src),
            HomSpaceInner::FermionParityU1SU2Irrep(dst),
        ) => tree_braider_hom!(
            src,
            dst,
            &p_codomain,
            &p_domain,
            &levels_codomain,
            &levels_domain
        ),
        _ => Err(PyValueError::new_err(
            "src and dst HomSpace sector families must match",
        )),
    }?;

    Ok(PyTreeTransformer { inner })
}

#[pyfunction]
pub(crate) fn tree_transposer(
    src: PyRef<'_, PyHomSpace>,
    dst: PyRef<'_, PyHomSpace>,
    p_codomain: Vec<usize>,
    p_domain: Vec<usize>,
) -> PyResult<PyTreeTransformer> {
    let inner = match (src.inner(), dst.inner()) {
        (HomSpaceInner::U1Irrep(src), HomSpaceInner::U1Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::SU2Irrep(src), HomSpaceInner::SU2Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::FermionParity(src), HomSpaceInner::FermionParity(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::Z2Irrep(src), HomSpaceInner::Z2Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::Z3Irrep(src), HomSpaceInner::Z3Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::Z4Irrep(src), HomSpaceInner::Z4Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::U1IrrepFermionParity(src), HomSpaceInner::U1IrrepFermionParity(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::FermionParityU1Irrep(src), HomSpaceInner::FermionParityU1Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::U1SU2Irrep(src), HomSpaceInner::U1SU2Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (HomSpaceInner::FermionParitySU2Irrep(src), HomSpaceInner::FermionParitySU2Irrep(dst)) => {
            tree_transposer_hom!(src, dst, &p_codomain, &p_domain)
        }
        (
            HomSpaceInner::FermionParityU1SU2Irrep(src),
            HomSpaceInner::FermionParityU1SU2Irrep(dst),
        ) => tree_transposer_hom!(src, dst, &p_codomain, &p_domain),
        _ => Err(PyValueError::new_err(
            "src and dst HomSpace sector families must match",
        )),
    }?;

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
