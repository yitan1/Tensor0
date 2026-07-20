use pyo3::prelude::*;
use pyo3::types::PyAny;
use tensor0_core::sector::SectorSpec as CoreSectorSpec;

use crate::pyconv::{core_err, py_hash, sector_key_from_py, sector_spec_static_key};

#[pyclass(name = "SectorSpec", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PySectorSpec {
    pub(crate) inner: CoreSectorSpec,
}

#[pymethods]
impl PySectorSpec {
    #[getter]
    fn static_key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        sector_spec_static_key(py, &self.inner)
    }

    fn __matmul__(&self, other: PyRef<'_, PySectorSpec>) -> PyResult<PySectorSpec> {
        Ok(PySectorSpec {
            inner: CoreSectorSpec::product(vec![self.inner.clone(), other.inner.clone()])
                .map_err(core_err)?,
        })
    }

    fn __eq__(&self, other: PyRef<'_, PySectorSpec>) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> isize {
        py_hash("SectorSpec", &self.inner)
    }

    fn quantum_dim(&self, sector: &Bound<'_, PyAny>) -> PyResult<usize> {
        let sector = sector_key_from_py(sector)?;
        self.inner.quantum_dim(&sector).map_err(core_err)
    }
}

fn add_sector_constant(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    name: &str,
    inner: CoreSectorSpec,
) -> PyResult<()> {
    module.add(name, Py::new(py, PySectorSpec { inner })?)
}

pub(crate) fn add_sector_constants(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_sector_constant(py, module, "Trivial", CoreSectorSpec::trivial())?;
    add_sector_constant(py, module, "U1Irrep", CoreSectorSpec::u1())?;
    add_sector_constant(py, module, "SU2Irrep", CoreSectorSpec::su2())?;
    add_sector_constant(
        py,
        module,
        "FermionParity",
        CoreSectorSpec::fermion_parity(),
    )?;
    add_sector_constant(
        py,
        module,
        "Z2Irrep",
        CoreSectorSpec::zn(2).map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "Z3Irrep",
        CoreSectorSpec::zn(3).map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "Z4Irrep",
        CoreSectorSpec::zn(4).map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "FermionNumber",
        CoreSectorSpec::product(vec![CoreSectorSpec::u1(), CoreSectorSpec::fermion_parity()])
            .map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "FermionParityU1Irrep",
        CoreSectorSpec::product(vec![CoreSectorSpec::fermion_parity(), CoreSectorSpec::u1()])
            .map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "U1SU2Irrep",
        CoreSectorSpec::product(vec![CoreSectorSpec::u1(), CoreSectorSpec::su2()])
            .map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "FermionParitySU2Irrep",
        CoreSectorSpec::product(vec![
            CoreSectorSpec::fermion_parity(),
            CoreSectorSpec::su2(),
        ])
        .map_err(core_err)?,
    )?;
    add_sector_constant(
        py,
        module,
        "FermionParityU1SU2Irrep",
        CoreSectorSpec::product(vec![
            CoreSectorSpec::fermion_parity(),
            CoreSectorSpec::u1(),
            CoreSectorSpec::su2(),
        ])
        .map_err(core_err)?,
    )?;
    Ok(())
}
