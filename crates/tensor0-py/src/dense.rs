use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyTuple};
use tensor0_core::dense::product_axes as core_product_axes;
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, SU2Irrep, Sector, U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::ProductSpace;

use crate::pyconv::{core_err, sector_key_from_py};
use crate::space::{ProductSpaceInner, PyProductSpace};

macro_rules! dispatch_product {
    ($product:expr, $handler:ident($($arg:expr),* $(,)?)) => {{
        match $product {
            ProductSpaceInner::U1Irrep(product) => $handler::<U1Irrep>(product, $($arg),*),
            ProductSpaceInner::SU2Irrep(product) => $handler::<SU2Irrep>(product, $($arg),*),
            ProductSpaceInner::FermionParity(product) => {
                $handler::<FermionParity>(product, $($arg),*)
            }
            ProductSpaceInner::Z2Irrep(product) => $handler::<Z2Irrep>(product, $($arg),*),
            ProductSpaceInner::Z3Irrep(product) => $handler::<Z3Irrep>(product, $($arg),*),
            ProductSpaceInner::Z4Irrep(product) => $handler::<Z4Irrep>(product, $($arg),*),
            ProductSpaceInner::U1IrrepFermionParity(product) => {
                $handler::<FermionNumber>(product, $($arg),*)
            }
            ProductSpaceInner::FermionParityU1Irrep(product) => {
                $handler::<FermionParityU1Irrep>(product, $($arg),*)
            }
            ProductSpaceInner::U1SU2Irrep(product) => {
                $handler::<U1SU2Irrep>(product, $($arg),*)
            }
            ProductSpaceInner::FermionParitySU2Irrep(product) => {
                $handler::<FermionParitySU2Irrep>(product, $($arg),*)
            }
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => {
                $handler::<FermionParityU1SU2Irrep>(product, $($arg),*)
            }
        }
    }};
}

#[pyfunction]
pub(crate) fn product_axes(
    py: Python<'_>,
    product: PyRef<'_, PyProductSpace>,
    sectors: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    dispatch_product!(product.inner(), product_axes_for(py, sectors))
}

#[pyfunction]
pub(crate) fn product_dims(
    py: Python<'_>,
    product: PyRef<'_, PyProductSpace>,
) -> PyResult<Py<PyAny>> {
    dispatch_product!(product.inner(), product_dims_for(py))
}

fn product_axes_for<I: Sector>(
    product: &ProductSpace<I>,
    py: Python<'_>,
    sectors: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let axes = core_product_axes(product, &decode_sectors::<I>(sectors)?).map_err(core_err)?;
    dense_axes_py(py, &axes)
}

fn product_dims_for<I: Sector>(product: &ProductSpace<I>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, product.dims())?.into_any().unbind())
}

fn decode_sectors<I: Sector>(sectors: &Bound<'_, PyAny>) -> PyResult<Vec<I>> {
    let iter = sectors
        .try_iter()
        .map_err(|_| PyTypeError::new_err("sectors must be iterable"))?;
    iter.map(|sector| {
        let sector = sector?;
        let key = sector_key_from_py(&sector)?;
        I::decode_value(&key).map_err(core_err)
    })
    .collect()
}

fn dense_axes_py(py: Python<'_>, axes: &[(usize, usize, usize, usize)]) -> PyResult<Py<PyAny>> {
    Ok(PyTuple::new(py, axes.iter().copied())?.into_any().unbind())
}
