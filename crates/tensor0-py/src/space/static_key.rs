use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::IntoPyObject;

use crate::pyconv::sector_spec_static_key;

use super::graded::{GradedSpaceInner, PyElementarySpace};

pub(super) fn space_static_key(py: Python<'_>, inner: &GradedSpaceInner) -> PyResult<Py<PyAny>> {
    let sector_key = sector_spec_static_key(py, &inner.sector_spec())?;
    let sectors = inner.sectors_py(py)?;
    let key = (
        "space",
        sector_key,
        sectors,
        inner.is_dual(),
        inner.fingerprint(),
    )
        .into_pyobject(py)?;
    Ok(key.into_any().unbind())
}

pub(super) fn space_static_keys_tuple(
    py: Python<'_>,
    spaces: Vec<GradedSpaceInner>,
) -> PyResult<Py<PyAny>> {
    let keys = spaces
        .iter()
        .map(|space| space_static_key(py, space))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, keys)?.into_any().unbind())
}

pub(super) fn spaces_tuple(py: Python<'_>, spaces: Vec<GradedSpaceInner>) -> PyResult<Py<PyAny>> {
    let spaces = spaces
        .into_iter()
        .map(|inner| Py::new(py, PyElementarySpace { inner }).map(|space| space.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, spaces)?.into_any().unbind())
}
