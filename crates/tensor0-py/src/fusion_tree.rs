use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use pyo3::IntoPyObject;
use tensor0_core::fusion_tree::{
    fusiontree_pair_tensor as core_fusiontree_pair_tensor,
    fusiontree_tensor as core_fusiontree_tensor, FusionTree, FusionTreePair,
};
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, SU2Irrep, Sector, U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};

use crate::pyconv::{
    arrayd_to_numpy, core_err, py_hash, sector_spec_static_key, sector_tuple_py, sectors_tuple_py,
    tuple_from_usizes,
};

#[pyclass(name = "FusionTree", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PyFusionTree {
    pub(crate) inner: FusionTreeInner,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum FusionTreeInner {
    U1Irrep(FusionTree<U1Irrep>),
    SU2Irrep(FusionTree<SU2Irrep>),
    FermionParity(FusionTree<FermionParity>),
    Z2Irrep(FusionTree<Z2Irrep>),
    Z3Irrep(FusionTree<Z3Irrep>),
    Z4Irrep(FusionTree<Z4Irrep>),
    U1IrrepFermionParity(FusionTree<FermionNumber>),
    FermionParityU1Irrep(FusionTree<FermionParityU1Irrep>),
    U1SU2Irrep(FusionTree<U1SU2Irrep>),
    FermionParitySU2Irrep(FusionTree<FermionParitySU2Irrep>),
    FermionParityU1SU2Irrep(FusionTree<FermionParityU1SU2Irrep>),
}

impl FusionTreeInner {
    fn tensor_py(&self, py: Python<'_>) -> PyResult<Py<PyArrayDyn<f64>>> {
        match self {
            FusionTreeInner::U1Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::SU2Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::FermionParity(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::Z2Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::Z3Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::Z4Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::U1IrrepFermionParity(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::FermionParityU1Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::U1SU2Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::FermionParitySU2Irrep(tree) => fusiontree_tensor_for(py, tree),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => fusiontree_tensor_for(py, tree),
        }
    }

    fn pair_tensor_py(
        py: Python<'_>,
        row: &FusionTreeInner,
        col: &FusionTreeInner,
    ) -> PyResult<Py<PyArrayDyn<f64>>> {
        match (row, col) {
            (FusionTreeInner::U1Irrep(row), FusionTreeInner::U1Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (FusionTreeInner::SU2Irrep(row), FusionTreeInner::SU2Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (FusionTreeInner::FermionParity(row), FusionTreeInner::FermionParity(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (FusionTreeInner::Z2Irrep(row), FusionTreeInner::Z2Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (FusionTreeInner::Z3Irrep(row), FusionTreeInner::Z3Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (FusionTreeInner::Z4Irrep(row), FusionTreeInner::Z4Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (
                FusionTreeInner::U1IrrepFermionParity(row),
                FusionTreeInner::U1IrrepFermionParity(col),
            ) => fusiontree_pair_tensor_for(py, row, col),
            (
                FusionTreeInner::FermionParityU1Irrep(row),
                FusionTreeInner::FermionParityU1Irrep(col),
            ) => fusiontree_pair_tensor_for(py, row, col),
            (FusionTreeInner::U1SU2Irrep(row), FusionTreeInner::U1SU2Irrep(col)) => {
                fusiontree_pair_tensor_for(py, row, col)
            }
            (
                FusionTreeInner::FermionParitySU2Irrep(row),
                FusionTreeInner::FermionParitySU2Irrep(col),
            ) => fusiontree_pair_tensor_for(py, row, col),
            (
                FusionTreeInner::FermionParityU1SU2Irrep(row),
                FusionTreeInner::FermionParityU1SU2Irrep(col),
            ) => fusiontree_pair_tensor_for(py, row, col),
            _ => Err(PyValueError::new_err(
                "fusion tree pair sector families must match",
            )),
        }
    }

    fn uncoupled_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FusionTreeInner::U1Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::SU2Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::FermionParity(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::Z2Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::Z3Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::Z4Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::U1IrrepFermionParity(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::FermionParityU1Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::U1SU2Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::FermionParitySU2Irrep(tree) => sectors_tuple_py(py, tree.uncoupled()),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => {
                sectors_tuple_py(py, tree.uncoupled())
            }
        }
    }

    fn coupled_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FusionTreeInner::U1Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::SU2Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::FermionParity(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::Z2Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::Z3Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::Z4Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::U1IrrepFermionParity(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::FermionParityU1Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::U1SU2Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::FermionParitySU2Irrep(tree) => sector_tuple_py(py, tree.coupled()),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => sector_tuple_py(py, tree.coupled()),
        }
    }

    fn is_dual_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flags = match self {
            FusionTreeInner::U1Irrep(tree) => tree.is_dual(),
            FusionTreeInner::SU2Irrep(tree) => tree.is_dual(),
            FusionTreeInner::FermionParity(tree) => tree.is_dual(),
            FusionTreeInner::Z2Irrep(tree) => tree.is_dual(),
            FusionTreeInner::Z3Irrep(tree) => tree.is_dual(),
            FusionTreeInner::Z4Irrep(tree) => tree.is_dual(),
            FusionTreeInner::U1IrrepFermionParity(tree) => tree.is_dual(),
            FusionTreeInner::FermionParityU1Irrep(tree) => tree.is_dual(),
            FusionTreeInner::U1SU2Irrep(tree) => tree.is_dual(),
            FusionTreeInner::FermionParitySU2Irrep(tree) => tree.is_dual(),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => tree.is_dual(),
        };
        Ok(PyTuple::new(py, flags.iter().copied())?.into_any().unbind())
    }

    fn innerlines_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FusionTreeInner::U1Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::SU2Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::FermionParity(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::Z2Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::Z3Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::Z4Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::U1IrrepFermionParity(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::FermionParityU1Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::U1SU2Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::FermionParitySU2Irrep(tree) => sectors_tuple_py(py, tree.innerlines()),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => {
                sectors_tuple_py(py, tree.innerlines())
            }
        }
    }

    fn vertices_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let vertices = match self {
            FusionTreeInner::U1Irrep(tree) => tree.vertices(),
            FusionTreeInner::SU2Irrep(tree) => tree.vertices(),
            FusionTreeInner::FermionParity(tree) => tree.vertices(),
            FusionTreeInner::Z2Irrep(tree) => tree.vertices(),
            FusionTreeInner::Z3Irrep(tree) => tree.vertices(),
            FusionTreeInner::Z4Irrep(tree) => tree.vertices(),
            FusionTreeInner::U1IrrepFermionParity(tree) => tree.vertices(),
            FusionTreeInner::FermionParityU1Irrep(tree) => tree.vertices(),
            FusionTreeInner::U1SU2Irrep(tree) => tree.vertices(),
            FusionTreeInner::FermionParitySU2Irrep(tree) => tree.vertices(),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => tree.vertices(),
        };
        tuple_from_usizes(py, vertices)
    }

    fn static_key_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            FusionTreeInner::U1Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::SU2Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::FermionParity(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::Z2Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::Z3Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::Z4Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::U1IrrepFermionParity(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::FermionParityU1Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::U1SU2Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::FermionParitySU2Irrep(tree) => fusiontree_static_key_py(py, tree),
            FusionTreeInner::FermionParityU1SU2Irrep(tree) => fusiontree_static_key_py(py, tree),
        }
    }
}

#[pyfunction]
pub(crate) fn fusiontree_tensor(
    py: Python<'_>,
    tree: PyRef<'_, PyFusionTree>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    tree.inner.tensor_py(py)
}

#[pyfunction]
pub(crate) fn fusiontree_pair_tensor(
    py: Python<'_>,
    row: PyRef<'_, PyFusionTree>,
    col: PyRef<'_, PyFusionTree>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    FusionTreeInner::pair_tensor_py(py, &row.inner, &col.inner)
}

#[pymethods]
impl PyFusionTree {
    #[getter]
    fn uncoupled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.uncoupled_py(py)
    }

    #[getter]
    fn coupled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.coupled_py(py)
    }

    #[getter]
    fn is_dual(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.is_dual_py(py)
    }

    #[getter]
    fn innerlines(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.innerlines_py(py)
    }

    #[getter]
    fn vertices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.vertices_py(py)
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.static_key_py(py)
    }

    fn __eq__(&self, other: PyRef<'_, PyFusionTree>) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        py_hash("FusionTree", &self.inner)
    }
}

fn fusiontree_tensor_for<I: Sector>(
    py: Python<'_>,
    tree: &FusionTree<I>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let tensor = core_fusiontree_tensor(tree).map_err(core_err)?;
    arrayd_to_numpy(py, tensor)
}

fn fusiontree_pair_tensor_for<I: Sector>(
    py: Python<'_>,
    row: &FusionTree<I>,
    col: &FusionTree<I>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let tensor = core_fusiontree_pair_tensor(row, col).map_err(core_err)?;
    arrayd_to_numpy(py, tensor)
}

pub(crate) fn fusiontree_pairs_py<'a, I, P, F>(
    py: Python<'_>,
    pairs: P,
    wrap: F,
) -> PyResult<Py<PyAny>>
where
    I: Sector + 'a,
    P: IntoIterator<Item = &'a FusionTreePair<I>>,
    F: Fn(FusionTree<I>) -> FusionTreeInner + Copy,
{
    let items = pairs
        .into_iter()
        .map(|pair| fusiontree_pair_py(py, pair, wrap))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, items)?.into_any().unbind())
}

pub(crate) fn fusiontree_pair_py<I, F>(
    py: Python<'_>,
    pair: &FusionTreePair<I>,
    wrap: F,
) -> PyResult<Py<PyAny>>
where
    I: Sector,
    F: Fn(FusionTree<I>) -> FusionTreeInner + Copy,
{
    let row = Py::new(
        py,
        PyFusionTree {
            inner: wrap(pair.row.clone()),
        },
    )?
    .into_any();
    let col = Py::new(
        py,
        PyFusionTree {
            inner: wrap(pair.col.clone()),
        },
    )?
    .into_any();
    Ok(PyTuple::new(py, [row, col])?.into_any().unbind())
}

pub(crate) fn fusiontree_static_key_py<I: Sector>(
    py: Python<'_>,
    tree: &FusionTree<I>,
) -> PyResult<Py<PyAny>> {
    let sector_spec = sector_spec_static_key(py, &I::sector_spec())?;
    let uncoupled = sectors_tuple_py(py, tree.uncoupled())?;
    let coupled = sector_tuple_py(py, tree.coupled())?;
    let is_dual = PyTuple::new(py, tree.is_dual().iter().copied())?
        .into_any()
        .unbind();
    let innerlines = sectors_tuple_py(py, tree.innerlines())?;
    let vertices = tuple_from_usizes(py, tree.vertices())?;
    let key = (
        "fusiontree",
        sector_spec,
        uncoupled,
        coupled,
        is_dual,
        innerlines,
        vertices,
    )
        .into_pyobject(py)?;
    Ok(key.into_any().unbind())
}
