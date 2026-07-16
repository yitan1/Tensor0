use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::IntoPyObject;
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, SU2Irrep, U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::{HomSpace as CoreHomSpace, HomSpaceSpec};

use crate::pyconv::{core_err, py_hash, sector_spec_static_key};

use super::graded::{GradedSpaceInner, PyElementarySpace};
use super::product::{ProductSpaceInner, PyProductSpace};
use super::static_key::{space_static_keys_tuple, spaces_tuple};

#[pyclass(name = "HomSpace", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyHomSpace {
    inner: HomSpaceInner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HomSpaceInner {
    U1Irrep(CoreHomSpace<U1Irrep>),
    SU2Irrep(CoreHomSpace<SU2Irrep>),
    FermionParity(CoreHomSpace<FermionParity>),
    Z2Irrep(CoreHomSpace<Z2Irrep>),
    Z3Irrep(CoreHomSpace<Z3Irrep>),
    Z4Irrep(CoreHomSpace<Z4Irrep>),
    U1IrrepFermionParity(CoreHomSpace<FermionNumber>),
    FermionParityU1Irrep(CoreHomSpace<FermionParityU1Irrep>),
    U1SU2Irrep(CoreHomSpace<U1SU2Irrep>),
    FermionParitySU2Irrep(CoreHomSpace<FermionParitySU2Irrep>),
    FermionParityU1SU2Irrep(CoreHomSpace<FermionParityU1SU2Irrep>),
}

macro_rules! build_hom_from_products {
    ($variant:ident, $sector:ty, $codomain:expr, $domain:expr) => {{
        match ($codomain, $domain) {
            (ProductSpaceInner::$variant(codomain), ProductSpaceInner::$variant(domain)) => Ok(
                HomSpaceInner::$variant(CoreHomSpace::<$sector>::new(codomain, domain)),
            ),
            _ => Err(PyValueError::new_err(
                "hom product spaces must have the same sector family",
            )),
        }
    }};
}

macro_rules! dispatch_hom_method {
    ($variant:ident, $hom:expr, $method:ident $(, $argument:expr)*) => {
        $hom.$method($($argument),*)
            .map(HomSpaceInner::$variant)
            .map_err(core_err)
    };
}

macro_rules! visible_leg_hom {
    ($variant:ident, $hom:expr, $index:expr) => {
        $hom.visible_leg($index)
            .map(GradedSpaceInner::$variant)
            .map_err(core_err)
    };
}

macro_rules! visible_legs_hom {
    ($variant:ident, $hom:expr) => {
        $hom.visible_legs()
            .into_iter()
            .map(GradedSpaceInner::$variant)
            .collect()
    };
}

macro_rules! dispatch_homspace_method {
    ($hom:expr, $method:ident $(, $argument:expr)*) => {
        match $hom {
            HomSpaceInner::U1Irrep(hom) => {
                dispatch_hom_method!(U1Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::SU2Irrep(hom) => {
                dispatch_hom_method!(SU2Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::FermionParity(hom) => {
                dispatch_hom_method!(FermionParity, hom, $method $(, $argument)*)
            }
            HomSpaceInner::Z2Irrep(hom) => {
                dispatch_hom_method!(Z2Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::Z3Irrep(hom) => {
                dispatch_hom_method!(Z3Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::Z4Irrep(hom) => {
                dispatch_hom_method!(Z4Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::U1IrrepFermionParity(hom) => {
                dispatch_hom_method!(U1IrrepFermionParity, hom, $method $(, $argument)*)
            }
            HomSpaceInner::FermionParityU1Irrep(hom) => {
                dispatch_hom_method!(FermionParityU1Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::U1SU2Irrep(hom) => {
                dispatch_hom_method!(U1SU2Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::FermionParitySU2Irrep(hom) => {
                dispatch_hom_method!(FermionParitySU2Irrep, hom, $method $(, $argument)*)
            }
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => {
                dispatch_hom_method!(FermionParityU1SU2Irrep, hom, $method $(, $argument)*)
            }
        }
    };
}

#[pyfunction]
pub(crate) fn make_hom_products(
    codomain: PyRef<'_, PyProductSpace>,
    domain: PyRef<'_, PyProductSpace>,
) -> PyResult<PyHomSpace> {
    Ok(PyHomSpace {
        inner: HomSpaceInner::from_products(codomain.inner().clone(), domain.inner().clone())?,
    })
}

impl PyHomSpace {
    pub(crate) fn inner(&self) -> &HomSpaceInner {
        &self.inner
    }
}

#[pymethods]
impl PyHomSpace {
    #[getter]
    fn codomain(&self) -> PyProductSpace {
        PyProductSpace {
            inner: self.inner.codomain_product(),
        }
    }

    #[getter]
    fn domain(&self) -> PyProductSpace {
        PyProductSpace {
            inner: self.inner.domain_product(),
        }
    }

    #[getter]
    fn numout(&self) -> usize {
        self.inner.numout()
    }

    #[getter]
    fn numin(&self) -> usize {
        self.inner.numin()
    }

    #[getter]
    fn numind(&self) -> usize {
        self.inner.numind()
    }

    #[getter]
    fn visible_legs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        spaces_tuple(py, self.inner.visible_legs())
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sector_key = sector_spec_static_key(py, &self.inner.codomain_product().sector_spec())?;
        let codomain_keys = space_static_keys_tuple(py, self.inner.codomain_spaces())?;
        let domain_keys = space_static_keys_tuple(py, self.inner.domain_spaces())?;
        let key = ("hom", sector_key, codomain_keys, domain_keys).into_pyobject(py)?;
        Ok(key.into_any().unbind())
    }

    fn permute(&self, p_codomain: Vec<usize>, p_domain: Vec<usize>) -> PyResult<PyHomSpace> {
        Ok(PyHomSpace {
            inner: self.inner.permute(&p_codomain, &p_domain)?,
        })
    }

    fn flip(&self, indices: Vec<usize>) -> PyResult<PyHomSpace> {
        Ok(PyHomSpace {
            inner: self.inner.flip(&indices)?,
        })
    }

    fn insert_left_unit(&self, position: isize, dual: bool) -> PyResult<PyHomSpace> {
        let position = nonnegative_unit_index(position, "unit insertion boundary")?;
        Ok(PyHomSpace {
            inner: self.inner.insert_left_unit(position, dual)?,
        })
    }

    fn insert_right_unit(&self, position: isize, dual: bool) -> PyResult<PyHomSpace> {
        let position = nonnegative_unit_index(position, "unit insertion boundary")?;
        Ok(PyHomSpace {
            inner: self.inner.insert_right_unit(position, dual)?,
        })
    }

    fn remove_unit(&self, index: isize) -> PyResult<PyHomSpace> {
        let index = nonnegative_unit_index(index, "unit removal index")?;
        Ok(PyHomSpace {
            inner: self.inner.remove_unit(index)?,
        })
    }

    fn __len__(&self) -> usize {
        self.inner.numind()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.inner.numind() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("visible index out of range"));
        }
        let inner = self.inner.visible_leg(index as usize)?;
        Py::new(py, PyElementarySpace { inner }).map(|space| space.into_any())
    }

    fn __eq__(&self, other: PyRef<'_, PyHomSpace>) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        py_hash("HomSpace", &self.inner.to_spec())
    }
}

fn nonnegative_unit_index(index: isize, context: &str) -> PyResult<usize> {
    usize::try_from(index)
        .map_err(|_| PyValueError::new_err(format!("{context} must be non-negative")))
}

impl HomSpaceInner {
    fn from_products(
        codomain: ProductSpaceInner,
        domain: ProductSpaceInner,
    ) -> PyResult<HomSpaceInner> {
        match codomain {
            ProductSpaceInner::U1Irrep(codomain) => {
                build_hom_from_products!(
                    U1Irrep,
                    U1Irrep,
                    ProductSpaceInner::U1Irrep(codomain),
                    domain
                )
            }
            ProductSpaceInner::SU2Irrep(codomain) => {
                build_hom_from_products!(
                    SU2Irrep,
                    SU2Irrep,
                    ProductSpaceInner::SU2Irrep(codomain),
                    domain
                )
            }
            ProductSpaceInner::FermionParity(codomain) => {
                build_hom_from_products!(
                    FermionParity,
                    FermionParity,
                    ProductSpaceInner::FermionParity(codomain),
                    domain
                )
            }
            ProductSpaceInner::Z2Irrep(codomain) => {
                build_hom_from_products!(
                    Z2Irrep,
                    Z2Irrep,
                    ProductSpaceInner::Z2Irrep(codomain),
                    domain
                )
            }
            ProductSpaceInner::Z3Irrep(codomain) => {
                build_hom_from_products!(
                    Z3Irrep,
                    Z3Irrep,
                    ProductSpaceInner::Z3Irrep(codomain),
                    domain
                )
            }
            ProductSpaceInner::Z4Irrep(codomain) => {
                build_hom_from_products!(
                    Z4Irrep,
                    Z4Irrep,
                    ProductSpaceInner::Z4Irrep(codomain),
                    domain
                )
            }
            ProductSpaceInner::U1IrrepFermionParity(codomain) => build_hom_from_products!(
                U1IrrepFermionParity,
                FermionNumber,
                ProductSpaceInner::U1IrrepFermionParity(codomain),
                domain
            ),
            ProductSpaceInner::FermionParityU1Irrep(codomain) => build_hom_from_products!(
                FermionParityU1Irrep,
                FermionParityU1Irrep,
                ProductSpaceInner::FermionParityU1Irrep(codomain),
                domain
            ),
            ProductSpaceInner::U1SU2Irrep(codomain) => build_hom_from_products!(
                U1SU2Irrep,
                U1SU2Irrep,
                ProductSpaceInner::U1SU2Irrep(codomain),
                domain
            ),
            ProductSpaceInner::FermionParitySU2Irrep(codomain) => build_hom_from_products!(
                FermionParitySU2Irrep,
                FermionParitySU2Irrep,
                ProductSpaceInner::FermionParitySU2Irrep(codomain),
                domain
            ),
            ProductSpaceInner::FermionParityU1SU2Irrep(codomain) => build_hom_from_products!(
                FermionParityU1SU2Irrep,
                FermionParityU1SU2Irrep,
                ProductSpaceInner::FermionParityU1SU2Irrep(codomain),
                domain
            ),
        }
    }

    fn permute(&self, p_codomain: &[usize], p_domain: &[usize]) -> PyResult<HomSpaceInner> {
        dispatch_homspace_method!(self, permute, p_codomain, p_domain)
    }

    fn flip(&self, indices: &[usize]) -> PyResult<HomSpaceInner> {
        dispatch_homspace_method!(self, flip, indices)
    }

    fn insert_left_unit(&self, position: usize, dual: bool) -> PyResult<HomSpaceInner> {
        dispatch_homspace_method!(self, insert_left_unit, position, dual)
    }

    fn insert_right_unit(&self, position: usize, dual: bool) -> PyResult<HomSpaceInner> {
        dispatch_homspace_method!(self, insert_right_unit, position, dual)
    }

    fn remove_unit(&self, index: usize) -> PyResult<HomSpaceInner> {
        dispatch_homspace_method!(self, remove_unit, index)
    }

    fn codomain_spaces(&self) -> Vec<GradedSpaceInner> {
        self.codomain_product().spaces()
    }

    fn numout(&self) -> usize {
        match self {
            HomSpaceInner::U1Irrep(hom) => hom.numout(),
            HomSpaceInner::SU2Irrep(hom) => hom.numout(),
            HomSpaceInner::FermionParity(hom) => hom.numout(),
            HomSpaceInner::Z2Irrep(hom) => hom.numout(),
            HomSpaceInner::Z3Irrep(hom) => hom.numout(),
            HomSpaceInner::Z4Irrep(hom) => hom.numout(),
            HomSpaceInner::U1IrrepFermionParity(hom) => hom.numout(),
            HomSpaceInner::FermionParityU1Irrep(hom) => hom.numout(),
            HomSpaceInner::U1SU2Irrep(hom) => hom.numout(),
            HomSpaceInner::FermionParitySU2Irrep(hom) => hom.numout(),
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => hom.numout(),
        }
    }

    fn numin(&self) -> usize {
        match self {
            HomSpaceInner::U1Irrep(hom) => hom.numin(),
            HomSpaceInner::SU2Irrep(hom) => hom.numin(),
            HomSpaceInner::FermionParity(hom) => hom.numin(),
            HomSpaceInner::Z2Irrep(hom) => hom.numin(),
            HomSpaceInner::Z3Irrep(hom) => hom.numin(),
            HomSpaceInner::Z4Irrep(hom) => hom.numin(),
            HomSpaceInner::U1IrrepFermionParity(hom) => hom.numin(),
            HomSpaceInner::FermionParityU1Irrep(hom) => hom.numin(),
            HomSpaceInner::U1SU2Irrep(hom) => hom.numin(),
            HomSpaceInner::FermionParitySU2Irrep(hom) => hom.numin(),
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => hom.numin(),
        }
    }

    fn numind(&self) -> usize {
        self.numout() + self.numin()
    }

    fn visible_leg(&self, index: usize) -> PyResult<GradedSpaceInner> {
        match self {
            HomSpaceInner::U1Irrep(hom) => visible_leg_hom!(U1Irrep, hom, index),
            HomSpaceInner::SU2Irrep(hom) => visible_leg_hom!(SU2Irrep, hom, index),
            HomSpaceInner::FermionParity(hom) => visible_leg_hom!(FermionParity, hom, index),
            HomSpaceInner::Z2Irrep(hom) => visible_leg_hom!(Z2Irrep, hom, index),
            HomSpaceInner::Z3Irrep(hom) => visible_leg_hom!(Z3Irrep, hom, index),
            HomSpaceInner::Z4Irrep(hom) => visible_leg_hom!(Z4Irrep, hom, index),
            HomSpaceInner::U1IrrepFermionParity(hom) => {
                visible_leg_hom!(U1IrrepFermionParity, hom, index)
            }
            HomSpaceInner::FermionParityU1Irrep(hom) => {
                visible_leg_hom!(FermionParityU1Irrep, hom, index)
            }
            HomSpaceInner::U1SU2Irrep(hom) => visible_leg_hom!(U1SU2Irrep, hom, index),
            HomSpaceInner::FermionParitySU2Irrep(hom) => {
                visible_leg_hom!(FermionParitySU2Irrep, hom, index)
            }
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => {
                visible_leg_hom!(FermionParityU1SU2Irrep, hom, index)
            }
        }
    }

    fn visible_legs(&self) -> Vec<GradedSpaceInner> {
        match self {
            HomSpaceInner::U1Irrep(hom) => visible_legs_hom!(U1Irrep, hom),
            HomSpaceInner::SU2Irrep(hom) => visible_legs_hom!(SU2Irrep, hom),
            HomSpaceInner::FermionParity(hom) => visible_legs_hom!(FermionParity, hom),
            HomSpaceInner::Z2Irrep(hom) => visible_legs_hom!(Z2Irrep, hom),
            HomSpaceInner::Z3Irrep(hom) => visible_legs_hom!(Z3Irrep, hom),
            HomSpaceInner::Z4Irrep(hom) => visible_legs_hom!(Z4Irrep, hom),
            HomSpaceInner::U1IrrepFermionParity(hom) => {
                visible_legs_hom!(U1IrrepFermionParity, hom)
            }
            HomSpaceInner::FermionParityU1Irrep(hom) => {
                visible_legs_hom!(FermionParityU1Irrep, hom)
            }
            HomSpaceInner::U1SU2Irrep(hom) => visible_legs_hom!(U1SU2Irrep, hom),
            HomSpaceInner::FermionParitySU2Irrep(hom) => {
                visible_legs_hom!(FermionParitySU2Irrep, hom)
            }
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => {
                visible_legs_hom!(FermionParityU1SU2Irrep, hom)
            }
        }
    }

    fn codomain_product(&self) -> ProductSpaceInner {
        match self {
            HomSpaceInner::U1Irrep(hom) => ProductSpaceInner::U1Irrep(hom.codomain().clone()),
            HomSpaceInner::SU2Irrep(hom) => ProductSpaceInner::SU2Irrep(hom.codomain().clone()),
            HomSpaceInner::FermionParity(hom) => {
                ProductSpaceInner::FermionParity(hom.codomain().clone())
            }
            HomSpaceInner::Z2Irrep(hom) => ProductSpaceInner::Z2Irrep(hom.codomain().clone()),
            HomSpaceInner::Z3Irrep(hom) => ProductSpaceInner::Z3Irrep(hom.codomain().clone()),
            HomSpaceInner::Z4Irrep(hom) => ProductSpaceInner::Z4Irrep(hom.codomain().clone()),
            HomSpaceInner::U1IrrepFermionParity(hom) => {
                ProductSpaceInner::U1IrrepFermionParity(hom.codomain().clone())
            }
            HomSpaceInner::FermionParityU1Irrep(hom) => {
                ProductSpaceInner::FermionParityU1Irrep(hom.codomain().clone())
            }
            HomSpaceInner::U1SU2Irrep(hom) => ProductSpaceInner::U1SU2Irrep(hom.codomain().clone()),
            HomSpaceInner::FermionParitySU2Irrep(hom) => {
                ProductSpaceInner::FermionParitySU2Irrep(hom.codomain().clone())
            }
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => {
                ProductSpaceInner::FermionParityU1SU2Irrep(hom.codomain().clone())
            }
        }
    }

    fn domain_spaces(&self) -> Vec<GradedSpaceInner> {
        self.domain_product().spaces()
    }

    fn domain_product(&self) -> ProductSpaceInner {
        match self {
            HomSpaceInner::U1Irrep(hom) => ProductSpaceInner::U1Irrep(hom.domain().clone()),
            HomSpaceInner::SU2Irrep(hom) => ProductSpaceInner::SU2Irrep(hom.domain().clone()),
            HomSpaceInner::FermionParity(hom) => {
                ProductSpaceInner::FermionParity(hom.domain().clone())
            }
            HomSpaceInner::Z2Irrep(hom) => ProductSpaceInner::Z2Irrep(hom.domain().clone()),
            HomSpaceInner::Z3Irrep(hom) => ProductSpaceInner::Z3Irrep(hom.domain().clone()),
            HomSpaceInner::Z4Irrep(hom) => ProductSpaceInner::Z4Irrep(hom.domain().clone()),
            HomSpaceInner::U1IrrepFermionParity(hom) => {
                ProductSpaceInner::U1IrrepFermionParity(hom.domain().clone())
            }
            HomSpaceInner::FermionParityU1Irrep(hom) => {
                ProductSpaceInner::FermionParityU1Irrep(hom.domain().clone())
            }
            HomSpaceInner::U1SU2Irrep(hom) => ProductSpaceInner::U1SU2Irrep(hom.domain().clone()),
            HomSpaceInner::FermionParitySU2Irrep(hom) => {
                ProductSpaceInner::FermionParitySU2Irrep(hom.domain().clone())
            }
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => {
                ProductSpaceInner::FermionParityU1SU2Irrep(hom.domain().clone())
            }
        }
    }

    fn to_spec(&self) -> HomSpaceSpec {
        match self {
            HomSpaceInner::U1Irrep(hom) => hom.to_spec(),
            HomSpaceInner::SU2Irrep(hom) => hom.to_spec(),
            HomSpaceInner::FermionParity(hom) => hom.to_spec(),
            HomSpaceInner::Z2Irrep(hom) => hom.to_spec(),
            HomSpaceInner::Z3Irrep(hom) => hom.to_spec(),
            HomSpaceInner::Z4Irrep(hom) => hom.to_spec(),
            HomSpaceInner::U1IrrepFermionParity(hom) => hom.to_spec(),
            HomSpaceInner::FermionParityU1Irrep(hom) => hom.to_spec(),
            HomSpaceInner::U1SU2Irrep(hom) => hom.to_spec(),
            HomSpaceInner::FermionParitySU2Irrep(hom) => hom.to_spec(),
            HomSpaceInner::FermionParityU1SU2Irrep(hom) => hom.to_spec(),
        }
    }
}
