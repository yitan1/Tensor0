use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, GroupSpec, SU2Irrep, Sector, SectorSpec as CoreSectorSpec, Trivial,
    U1Irrep, U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::{
    fuse_product_space as core_fuse_product_space, GradedSpace, ProductSpace,
};

use crate::pyconv::{core_err, py_hash};
use crate::sector_type::PySectorSpec;

use super::graded::{GradedSpaceInner, PyElementarySpace};
use super::static_key::{product_space_static_key, spaces_tuple};

#[pyclass(name = "ProductSpace", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyProductSpace {
    pub(super) inner: ProductSpaceInner,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ProductSpaceInner {
    Trivial(ProductSpace<Trivial>),
    U1Irrep(ProductSpace<U1Irrep>),
    SU2Irrep(ProductSpace<SU2Irrep>),
    FermionParity(ProductSpace<FermionParity>),
    Z2Irrep(ProductSpace<Z2Irrep>),
    Z3Irrep(ProductSpace<Z3Irrep>),
    Z4Irrep(ProductSpace<Z4Irrep>),
    U1IrrepFermionParity(ProductSpace<FermionNumber>),
    FermionParityU1Irrep(ProductSpace<FermionParityU1Irrep>),
    U1SU2Irrep(ProductSpace<U1SU2Irrep>),
    FermionParitySU2Irrep(ProductSpace<FermionParitySU2Irrep>),
    FermionParityU1SU2Irrep(ProductSpace<FermionParityU1SU2Irrep>),
}

macro_rules! build_product_space {
    ($variant:ident, $sector:ty, $spaces:expr) => {{
        let factors = collect_product_factors!($variant, $spaces)?;
        Ok(ProductSpaceInner::$variant(ProductSpace::<$sector>::new(
            factors,
        )))
    }};
}

macro_rules! collect_product_factors {
    ($variant:ident, $spaces:expr) => {{
        $spaces
            .into_iter()
            .map(|space| match space.inner {
                GradedSpaceInner::$variant(space) => Ok(space),
                _ => Err(PyValueError::new_err(
                    "all product space factors must have the same sector family",
                )),
            })
            .collect::<PyResult<Vec<_>>>()
    }};
}

#[pyfunction]
pub(crate) fn make_product_space(
    sector_spec: PyRef<'_, PySectorSpec>,
    spaces: &Bound<'_, PyAny>,
) -> PyResult<PyProductSpace> {
    Ok(PyProductSpace {
        inner: ProductSpaceInner::from_spaces_for_spec(
            &sector_spec.inner,
            extract_spaces(spaces)?,
        )?,
    })
}

#[pyfunction]
pub(crate) fn fuse(product: PyRef<'_, PyProductSpace>) -> PyResult<PyElementarySpace> {
    product
        .inner
        .fuse()
        .map(|inner| PyElementarySpace { inner })
}

impl PyProductSpace {
    pub(crate) fn inner(&self) -> &ProductSpaceInner {
        &self.inner
    }
}

#[pymethods]
impl PyProductSpace {
    #[getter]
    fn sector_spec(&self) -> PySectorSpec {
        PySectorSpec {
            inner: self.inner.sector_spec(),
        }
    }

    #[getter]
    fn spaces(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        spaces_tuple(py, self.inner.spaces())
    }

    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        product_space_static_key(py, &self.inner)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.inner.len() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 {
            return Err(PyIndexError::new_err("product space index out of range"));
        }
        let inner = self
            .inner
            .space_at(index as usize)
            .ok_or_else(|| PyIndexError::new_err("product space index out of range"))?;
        Py::new(py, PyElementarySpace { inner }).map(|space| space.into_any())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_product) = other.extract::<PyRef<'_, PyProductSpace>>() {
            return Ok(self.inner == other_product.inner);
        }
        Ok(false)
    }

    fn __hash__(&self) -> isize {
        py_hash("ProductSpace", &self.inner)
    }
}

impl ProductSpaceInner {
    pub(super) fn from_spaces_for_spec(
        spec: &CoreSectorSpec,
        spaces: Vec<PyElementarySpace>,
    ) -> PyResult<ProductSpaceInner> {
        match spec.clone().canonicalize().map_err(core_err)? {
            CoreSectorSpec::Trivial => build_product_space!(Trivial, Trivial, spaces),
            CoreSectorSpec::Irrep {
                group: GroupSpec::U1,
            } => build_product_space!(U1Irrep, U1Irrep, spaces),
            CoreSectorSpec::Irrep {
                group: GroupSpec::SU2,
            } => build_product_space!(SU2Irrep, SU2Irrep, spaces),
            CoreSectorSpec::FermionParity => {
                build_product_space!(FermionParity, FermionParity, spaces)
            }
            CoreSectorSpec::Irrep {
                group: GroupSpec::ZN { n: 2 },
            } => build_product_space!(Z2Irrep, Z2Irrep, spaces),
            CoreSectorSpec::Irrep {
                group: GroupSpec::ZN { n: 3 },
            } => build_product_space!(Z3Irrep, Z3Irrep, spaces),
            CoreSectorSpec::Irrep {
                group: GroupSpec::ZN { n: 4 },
            } => build_product_space!(Z4Irrep, Z4Irrep, spaces),
            CoreSectorSpec::Product { components }
                if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::fermion_parity()] =>
            {
                build_product_space!(U1IrrepFermionParity, FermionNumber, spaces)
            }
            CoreSectorSpec::Product { components }
                if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::u1()] =>
            {
                build_product_space!(FermionParityU1Irrep, FermionParityU1Irrep, spaces)
            }
            CoreSectorSpec::Product { components }
                if components == vec![CoreSectorSpec::u1(), CoreSectorSpec::su2()] =>
            {
                build_product_space!(U1SU2Irrep, U1SU2Irrep, spaces)
            }
            CoreSectorSpec::Product { components }
                if components == vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::su2()] =>
            {
                build_product_space!(FermionParitySU2Irrep, FermionParitySU2Irrep, spaces)
            }
            CoreSectorSpec::Product { components }
                if components
                    == vec![
                        CoreSectorSpec::fermion_parity(),
                        CoreSectorSpec::u1(),
                        CoreSectorSpec::su2(),
                    ] =>
            {
                build_product_space!(FermionParityU1SU2Irrep, FermionParityU1SU2Irrep, spaces)
            }
            _ => Err(PyValueError::new_err("unsupported sector spec")),
        }
    }

    pub(super) fn sector_spec(&self) -> CoreSectorSpec {
        match self {
            ProductSpaceInner::Trivial(_) => Trivial::sector_spec(),
            ProductSpaceInner::U1Irrep(_) => U1Irrep::sector_spec(),
            ProductSpaceInner::SU2Irrep(_) => SU2Irrep::sector_spec(),
            ProductSpaceInner::FermionParity(_) => FermionParity::sector_spec(),
            ProductSpaceInner::Z2Irrep(_) => Z2Irrep::sector_spec(),
            ProductSpaceInner::Z3Irrep(_) => Z3Irrep::sector_spec(),
            ProductSpaceInner::Z4Irrep(_) => Z4Irrep::sector_spec(),
            ProductSpaceInner::U1IrrepFermionParity(_) => FermionNumber::sector_spec(),
            ProductSpaceInner::FermionParityU1Irrep(_) => FermionParityU1Irrep::sector_spec(),
            ProductSpaceInner::U1SU2Irrep(_) => U1SU2Irrep::sector_spec(),
            ProductSpaceInner::FermionParitySU2Irrep(_) => FermionParitySU2Irrep::sector_spec(),
            ProductSpaceInner::FermionParityU1SU2Irrep(_) => FermionParityU1SU2Irrep::sector_spec(),
        }
    }

    pub(super) fn spaces(&self) -> Vec<GradedSpaceInner> {
        match self {
            ProductSpaceInner::Trivial(product) => {
                factors_to_spaces(product, GradedSpaceInner::Trivial)
            }
            ProductSpaceInner::U1Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::U1Irrep)
            }
            ProductSpaceInner::SU2Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::SU2Irrep)
            }
            ProductSpaceInner::FermionParity(product) => {
                factors_to_spaces(product, GradedSpaceInner::FermionParity)
            }
            ProductSpaceInner::Z2Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::Z2Irrep)
            }
            ProductSpaceInner::Z3Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::Z3Irrep)
            }
            ProductSpaceInner::Z4Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::Z4Irrep)
            }
            ProductSpaceInner::U1IrrepFermionParity(product) => {
                factors_to_spaces(product, GradedSpaceInner::U1IrrepFermionParity)
            }
            ProductSpaceInner::FermionParityU1Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::FermionParityU1Irrep)
            }
            ProductSpaceInner::U1SU2Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::U1SU2Irrep)
            }
            ProductSpaceInner::FermionParitySU2Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::FermionParitySU2Irrep)
            }
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => {
                factors_to_spaces(product, GradedSpaceInner::FermionParityU1SU2Irrep)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            ProductSpaceInner::Trivial(product) => product.factors().len(),
            ProductSpaceInner::U1Irrep(product) => product.factors().len(),
            ProductSpaceInner::SU2Irrep(product) => product.factors().len(),
            ProductSpaceInner::FermionParity(product) => product.factors().len(),
            ProductSpaceInner::Z2Irrep(product) => product.factors().len(),
            ProductSpaceInner::Z3Irrep(product) => product.factors().len(),
            ProductSpaceInner::Z4Irrep(product) => product.factors().len(),
            ProductSpaceInner::U1IrrepFermionParity(product) => product.factors().len(),
            ProductSpaceInner::FermionParityU1Irrep(product) => product.factors().len(),
            ProductSpaceInner::U1SU2Irrep(product) => product.factors().len(),
            ProductSpaceInner::FermionParitySU2Irrep(product) => product.factors().len(),
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => product.factors().len(),
        }
    }

    fn space_at(&self, index: usize) -> Option<GradedSpaceInner> {
        match self {
            ProductSpaceInner::Trivial(product) => {
                factor_at(product, index, GradedSpaceInner::Trivial)
            }
            ProductSpaceInner::U1Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::U1Irrep)
            }
            ProductSpaceInner::SU2Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::SU2Irrep)
            }
            ProductSpaceInner::FermionParity(product) => {
                factor_at(product, index, GradedSpaceInner::FermionParity)
            }
            ProductSpaceInner::Z2Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::Z2Irrep)
            }
            ProductSpaceInner::Z3Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::Z3Irrep)
            }
            ProductSpaceInner::Z4Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::Z4Irrep)
            }
            ProductSpaceInner::U1IrrepFermionParity(product) => {
                factor_at(product, index, GradedSpaceInner::U1IrrepFermionParity)
            }
            ProductSpaceInner::FermionParityU1Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::FermionParityU1Irrep)
            }
            ProductSpaceInner::U1SU2Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::U1SU2Irrep)
            }
            ProductSpaceInner::FermionParitySU2Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::FermionParitySU2Irrep)
            }
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => {
                factor_at(product, index, GradedSpaceInner::FermionParityU1SU2Irrep)
            }
        }
    }

    fn fuse(&self) -> PyResult<GradedSpaceInner> {
        match self {
            ProductSpaceInner::Trivial(product) => fuse_product(product, GradedSpaceInner::Trivial),
            ProductSpaceInner::U1Irrep(product) => fuse_product(product, GradedSpaceInner::U1Irrep),
            ProductSpaceInner::SU2Irrep(product) => {
                fuse_product(product, GradedSpaceInner::SU2Irrep)
            }
            ProductSpaceInner::FermionParity(product) => {
                fuse_product(product, GradedSpaceInner::FermionParity)
            }
            ProductSpaceInner::Z2Irrep(product) => fuse_product(product, GradedSpaceInner::Z2Irrep),
            ProductSpaceInner::Z3Irrep(product) => fuse_product(product, GradedSpaceInner::Z3Irrep),
            ProductSpaceInner::Z4Irrep(product) => fuse_product(product, GradedSpaceInner::Z4Irrep),
            ProductSpaceInner::U1IrrepFermionParity(product) => {
                fuse_product(product, GradedSpaceInner::U1IrrepFermionParity)
            }
            ProductSpaceInner::FermionParityU1Irrep(product) => {
                fuse_product(product, GradedSpaceInner::FermionParityU1Irrep)
            }
            ProductSpaceInner::U1SU2Irrep(product) => {
                fuse_product(product, GradedSpaceInner::U1SU2Irrep)
            }
            ProductSpaceInner::FermionParitySU2Irrep(product) => {
                fuse_product(product, GradedSpaceInner::FermionParitySU2Irrep)
            }
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => {
                fuse_product(product, GradedSpaceInner::FermionParityU1SU2Irrep)
            }
        }
    }
}

fn extract_spaces(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyElementarySpace>> {
    let spaces = obj.extract::<Vec<PyRef<'_, PyElementarySpace>>>()?;
    Ok(spaces.iter().map(|space| (*space).clone()).collect())
}

fn factor_at<I: Sector>(
    product: &ProductSpace<I>,
    index: usize,
    wrap: fn(GradedSpace<I>) -> GradedSpaceInner,
) -> Option<GradedSpaceInner> {
    product.get(index).cloned().map(wrap)
}

fn factors_to_spaces<I: Sector>(
    product: &ProductSpace<I>,
    wrap: fn(GradedSpace<I>) -> GradedSpaceInner,
) -> Vec<GradedSpaceInner> {
    product.factors().iter().cloned().map(wrap).collect()
}

fn fuse_product<I: Sector>(
    product: &ProductSpace<I>,
    wrap: fn(GradedSpace<I>) -> GradedSpaceInner,
) -> PyResult<GradedSpaceInner> {
    core_fuse_product_space(product).map(wrap).map_err(core_err)
}
