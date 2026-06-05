use pyo3::prelude::*;
use tensor0_core::sector::SectorSpec as CoreSectorSpec;

mod dense;
mod fusion_tree;
mod layout;
mod pyconv;
mod sector_type;
mod space;
mod transform;

use dense::{product_axes, product_dims};
use fusion_tree::{fusiontree_pair_tensor, fusiontree_tensor, PyFusionTree};
use layout::{
    _build_degeneracystructure_from_sectorstructure, build_degeneracystructure,
    build_sectorstructure, PyBlockStructure, PyDegeneracyStructure, PySectorStructure,
    PySubblockStructure,
};
use pyconv::core_err;
use sector_type::{add_sector_constant, PySectorSpec};
use space::{
    fuse, infimum_space, make_hom_products, make_product_space, make_space, PyElementarySpace,
    PyHomSpace, PyProductSpace,
};
use transform::{
    tree_braider, tree_transposer, PyAbelianTransformData, PyGenericTransformData,
    PyGenericTransformStructures, PyTreeTransformer,
};

#[pyfunction]
fn native_version() -> &'static str {
    tensor0_core::native_version()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PySectorSpec>()?;
    module.add_class::<PyElementarySpace>()?;
    module.add_class::<PyProductSpace>()?;
    module.add_class::<PyHomSpace>()?;
    module.add_class::<PySectorStructure>()?;
    module.add_class::<PyDegeneracyStructure>()?;
    module.add_class::<PyBlockStructure>()?;
    module.add_class::<PySubblockStructure>()?;
    module.add_class::<PyFusionTree>()?;
    module.add_class::<PyTreeTransformer>()?;
    module.add_class::<PyAbelianTransformData>()?;
    module.add_class::<PyGenericTransformData>()?;
    module.add_class::<PyGenericTransformStructures>()?;
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    module.add_function(wrap_pyfunction!(make_space, module)?)?;
    module.add_function(wrap_pyfunction!(make_product_space, module)?)?;
    module.add_function(wrap_pyfunction!(make_hom_products, module)?)?;
    module.add_function(wrap_pyfunction!(fuse, module)?)?;
    module.add_function(wrap_pyfunction!(infimum_space, module)?)?;
    module.add_function(wrap_pyfunction!(build_sectorstructure, module)?)?;
    module.add_function(wrap_pyfunction!(build_degeneracystructure, module)?)?;
    module.add_function(wrap_pyfunction!(
        _build_degeneracystructure_from_sectorstructure,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(fusiontree_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(fusiontree_pair_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(product_axes, module)?)?;
    module.add_function(wrap_pyfunction!(product_dims, module)?)?;
    module.add_function(wrap_pyfunction!(tree_braider, module)?)?;
    module.add_function(wrap_pyfunction!(tree_transposer, module)?)?;

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
