use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods};
use tensor0_core::sector::{
    FermionNumber, FermionParity, FermionParitySU2Irrep, FermionParityU1Irrep,
    FermionParityU1SU2Irrep, GroupSpec, SU2Irrep, Sector, SectorSpec as CoreSectorSpec, U1Irrep,
    U1SU2Irrep, Z2Irrep, Z3Irrep, Z4Irrep,
};
use tensor0_core::space::{
    fuse_product_space as core_fuse_product_space, ProductSpace, ProductSpaceSpec,
};

use crate::pyconv::{core_err, py_hash};
use crate::sector_type::PySectorSpec;

use super::graded::{GradedSpaceInner, PyElementarySpace};
use super::static_key::spaces_tuple;

#[pyclass(name = "ProductSpace", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PyProductSpace {
    pub(super) inner: ProductSpaceInner,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProductSpaceInner {
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
        ProductSpace::<$sector>::new(factors)
            .map(ProductSpaceInner::$variant)
            .map_err(core_err)
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
    let spec = sector_spec.inner.clone().canonicalize().map_err(core_err)?;
    Ok(PyProductSpace {
        inner: ProductSpaceInner::from_spaces_for_spec(&spec, extract_spaces(spaces)?)?,
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
    fn fingerprint(&self) -> u128 {
        self.inner.fingerprint()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let len = self.inner.len() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(PyIndexError::new_err("product space index out of range"));
        }
        let inner = self.inner.spaces()[index as usize].clone();
        Py::new(py, PyElementarySpace { inner }).map(|space| space.into_any())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_product) = other.extract::<PyRef<'_, PyProductSpace>>() {
            return Ok(self.inner == other_product.inner);
        }
        if let Ok(other_spaces) = other.extract::<Vec<PyRef<'_, PyElementarySpace>>>() {
            let own_spaces = self.inner.spaces();
            return Ok(own_spaces.len() == other_spaces.len()
                && own_spaces
                    .iter()
                    .zip(other_spaces.iter())
                    .all(|(left, right)| left == &right.inner));
        }
        Ok(false)
    }

    fn __hash__(&self) -> isize {
        py_hash("ProductSpace", &self.inner.to_spec())
    }
}

impl ProductSpaceInner {
    pub(super) fn from_spaces_for_spec(
        spec: &CoreSectorSpec,
        spaces: Vec<PyElementarySpace>,
    ) -> PyResult<ProductSpaceInner> {
        match spec.clone().canonicalize().map_err(core_err)? {
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

    fn sector_spec(&self) -> CoreSectorSpec {
        match self {
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
            ProductSpaceInner::U1Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::U1Irrep)
                .collect(),
            ProductSpaceInner::SU2Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::SU2Irrep)
                .collect(),
            ProductSpaceInner::FermionParity(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::FermionParity)
                .collect(),
            ProductSpaceInner::Z2Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::Z2Irrep)
                .collect(),
            ProductSpaceInner::Z3Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::Z3Irrep)
                .collect(),
            ProductSpaceInner::Z4Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::Z4Irrep)
                .collect(),
            ProductSpaceInner::U1IrrepFermionParity(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::U1IrrepFermionParity)
                .collect(),
            ProductSpaceInner::FermionParityU1Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::FermionParityU1Irrep)
                .collect(),
            ProductSpaceInner::U1SU2Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::U1SU2Irrep)
                .collect(),
            ProductSpaceInner::FermionParitySU2Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::FermionParitySU2Irrep)
                .collect(),
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => product
                .factors()
                .iter()
                .cloned()
                .map(GradedSpaceInner::FermionParityU1SU2Irrep)
                .collect(),
        }
    }

    fn len(&self) -> usize {
        match self {
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

    fn fingerprint(&self) -> u128 {
        match self {
            ProductSpaceInner::U1Irrep(product) => product.fingerprint(),
            ProductSpaceInner::SU2Irrep(product) => product.fingerprint(),
            ProductSpaceInner::FermionParity(product) => product.fingerprint(),
            ProductSpaceInner::Z2Irrep(product) => product.fingerprint(),
            ProductSpaceInner::Z3Irrep(product) => product.fingerprint(),
            ProductSpaceInner::Z4Irrep(product) => product.fingerprint(),
            ProductSpaceInner::U1IrrepFermionParity(product) => product.fingerprint(),
            ProductSpaceInner::FermionParityU1Irrep(product) => product.fingerprint(),
            ProductSpaceInner::U1SU2Irrep(product) => product.fingerprint(),
            ProductSpaceInner::FermionParitySU2Irrep(product) => product.fingerprint(),
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => product.fingerprint(),
        }
    }

    pub(super) fn to_spec(&self) -> ProductSpaceSpec {
        match self {
            ProductSpaceInner::U1Irrep(product) => product.to_spec(),
            ProductSpaceInner::SU2Irrep(product) => product.to_spec(),
            ProductSpaceInner::FermionParity(product) => product.to_spec(),
            ProductSpaceInner::Z2Irrep(product) => product.to_spec(),
            ProductSpaceInner::Z3Irrep(product) => product.to_spec(),
            ProductSpaceInner::Z4Irrep(product) => product.to_spec(),
            ProductSpaceInner::U1IrrepFermionParity(product) => product.to_spec(),
            ProductSpaceInner::FermionParityU1Irrep(product) => product.to_spec(),
            ProductSpaceInner::U1SU2Irrep(product) => product.to_spec(),
            ProductSpaceInner::FermionParitySU2Irrep(product) => product.to_spec(),
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => product.to_spec(),
        }
    }

    fn fuse(&self) -> PyResult<GradedSpaceInner> {
        match self {
            ProductSpaceInner::U1Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::U1Irrep)
                .map_err(core_err),
            ProductSpaceInner::SU2Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::SU2Irrep)
                .map_err(core_err),
            ProductSpaceInner::FermionParity(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::FermionParity)
                .map_err(core_err),
            ProductSpaceInner::Z2Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::Z2Irrep)
                .map_err(core_err),
            ProductSpaceInner::Z3Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::Z3Irrep)
                .map_err(core_err),
            ProductSpaceInner::Z4Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::Z4Irrep)
                .map_err(core_err),
            ProductSpaceInner::U1IrrepFermionParity(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::U1IrrepFermionParity)
                .map_err(core_err),
            ProductSpaceInner::FermionParityU1Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::FermionParityU1Irrep)
                .map_err(core_err),
            ProductSpaceInner::U1SU2Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::U1SU2Irrep)
                .map_err(core_err),
            ProductSpaceInner::FermionParitySU2Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::FermionParitySU2Irrep)
                .map_err(core_err),
            ProductSpaceInner::FermionParityU1SU2Irrep(product) => core_fuse_product_space(product)
                .map(GradedSpaceInner::FermionParityU1SU2Irrep)
                .map_err(core_err),
        }
    }
}

fn extract_spaces(obj: &Bound<'_, PyAny>) -> PyResult<Vec<PyElementarySpace>> {
    let spaces = obj.extract::<Vec<PyRef<'_, PyElementarySpace>>>()?;
    Ok(spaces.iter().map(|space| (*space).clone()).collect())
}
