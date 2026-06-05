use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::error::Tensor0Error;
use tensor0_core::sector::{GroupSpec, Sector, SectorSpec as CoreSectorSpec};

pub(crate) fn parse_sector_dims(
    sectors: &Bound<'_, PyAny>,
    expected_width: usize,
) -> PyResult<Vec<(Vec<i64>, usize)>> {
    let items = sectors
        .extract::<Vec<(Vec<i64>, i64)>>()
        .map_err(|_| PyTypeError::new_err("sectors must be normalized sector items"))?;

    items
        .into_iter()
        .map(|(sector, dim)| {
            require_width(sector.len(), expected_width)?;
            Ok((sector, parse_dim(dim)?))
        })
        .collect()
}

pub(crate) fn sector_key_from_py(sector: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Some(value) = sector_key_part_from_py(sector)? {
        return Ok(vec![value]);
    }
    let tuple = sector
        .cast::<PyTuple>()
        .map_err(|_| sector_key_type_error())?;
    tuple
        .iter()
        .map(|value| sector_key_part_from_py(&value)?.ok_or_else(sector_key_type_error))
        .collect()
}

pub(crate) fn sectors_py<I: Sector>(
    py: Python<'_>,
    space: &tensor0_core::space::GradedSpace<I>,
) -> PyResult<Py<PyAny>> {
    let pairs = space
        .sectors()
        .into_iter()
        .map(|(sector, dim)| {
            let encoded = sector.encode_value().into_iter().collect::<Vec<_>>();
            let sector_tuple = PyTuple::new(py, encoded)?;
            let pair = (sector_tuple, dim).into_pyobject(py)?;
            Ok(pair.into_any().unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, pairs)?.into_any().unbind())
}

pub(crate) fn sector_tuple_py<I: Sector>(py: Python<'_>, sector: &I) -> PyResult<Py<PyAny>> {
    let encoded = sector.encode_value().into_iter().collect::<Vec<_>>();
    Ok(PyTuple::new(py, encoded)?.into_any().unbind())
}

pub(crate) fn sectors_tuple_py<I: Sector>(py: Python<'_>, sectors: &[I]) -> PyResult<Py<PyAny>> {
    let items = sectors
        .iter()
        .map(|sector| sector_tuple_py(py, sector))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

pub(crate) fn tuple_from_usizes(py: Python<'_>, values: &[usize]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, values.iter().copied())?
        .into_any()
        .unbind())
}

pub(crate) fn sector_spec_static_key(py: Python<'_>, spec: &CoreSectorSpec) -> PyResult<Py<PyAny>> {
    match spec {
        CoreSectorSpec::Irrep {
            group: GroupSpec::U1,
        } => Ok(("irrep", ("u1",)).into_pyobject(py)?.into_any().unbind()),
        CoreSectorSpec::Irrep {
            group: GroupSpec::ZN { n },
        } => Ok(("irrep", ("zn", *n)).into_pyobject(py)?.into_any().unbind()),
        CoreSectorSpec::Irrep {
            group: GroupSpec::SU2,
        } => Ok(("irrep", ("su2",)).into_pyobject(py)?.into_any().unbind()),
        CoreSectorSpec::FermionParity => {
            Ok(("fermion_parity",).into_pyobject(py)?.into_any().unbind())
        }
        CoreSectorSpec::Product { components } => {
            let component_keys = components
                .iter()
                .map(|component| sector_spec_static_key(py, component))
                .collect::<PyResult<Vec<_>>>()?;
            let component_tuple = PyTuple::new(py, component_keys)?;
            let key = ("product", component_tuple).into_pyobject(py)?;
            Ok(key.into_any().unbind())
        }
    }
}

pub(crate) fn py_hash<T: Hash>(tag: &str, value: &T) -> isize {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    value.hash(&mut hasher);
    (hasher.finish() & (isize::MAX as u64)) as isize
}

pub(crate) fn core_err(err: Tensor0Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(crate) fn arrayd_to_numpy(
    py: Python<'_>,
    tensor: ArrayD<f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    Ok(tensor.into_pyarray(py).unbind())
}

fn require_width(actual: usize, expected: usize) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "sector value has width {actual}, expected {expected}"
        )))
    }
}

fn parse_dim(dim: i64) -> PyResult<usize> {
    usize::try_from(dim).map_err(|_| PyValueError::new_err("sector dimension must be non-negative"))
}

fn sector_key_part_from_py(value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.extract::<bool>().is_ok() {
        return Err(sector_key_type_error());
    }
    Ok(value.extract::<i64>().ok())
}

fn sector_key_type_error() -> PyErr {
    PyTypeError::new_err("sector key must be an int or a tuple of ints")
}
