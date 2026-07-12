use pyo3::prelude::*;

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
use sector_type::{add_sector_constants, PySectorSpec};
use space::{
    fuse, infimum_space, make_hom_products, make_product_space, make_space, PyElementarySpace,
    PyHomSpace, PyProductSpace,
};
use transform::{
    trace_transformer, tree_braider, tree_transposer, twist_is_trivial, twist_subblock_factors,
    PyAbelianTransformData, PyGenericTransformData, PyTreeTransformer,
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
    module.add_function(wrap_pyfunction!(twist_is_trivial, module)?)?;
    module.add_function(wrap_pyfunction!(twist_subblock_factors, module)?)?;
    module.add_function(wrap_pyfunction!(trace_transformer, module)?)?;
    module.add_function(wrap_pyfunction!(tree_braider, module)?)?;
    module.add_function(wrap_pyfunction!(tree_transposer, module)?)?;

    add_sector_constants(py, module)?;
    Ok(())
}
