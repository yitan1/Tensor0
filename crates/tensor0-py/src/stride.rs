#[cfg(tensor0_stride_ffi)]
use std::ffi::{c_char, c_void, CStr};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[cfg(tensor0_stride_ffi)]
unsafe extern "C" {
    fn Tensor0StrideGetWorkerLimit() -> u64;
    fn Tensor0StrideSetWorkerLimit(limit: u64);
    fn Tensor0StrideBuiltJaxVersion() -> *const c_char;
    fn Tensor0StrideBuiltJaxlibVersion() -> *const c_char;
    fn Tensor0StrideAccumulationInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationS32V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationF32V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationF16V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationBF16V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationC64V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationF64V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationC128V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationS64V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationU64V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationS16V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationS8V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationU8V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationU16V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationU32V1Handler() -> *mut c_void;
    fn Tensor0StrideAccumulationPredV1Handler() -> *mut c_void;
    fn Tensor0StrideDotInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideDotF32V1Handler() -> *mut c_void;
    fn Tensor0StrideDotF64V1Handler() -> *mut c_void;
    fn Tensor0StrideDotC64V1Handler() -> *mut c_void;
    fn Tensor0StrideDotC128V1Handler() -> *mut c_void;
    fn Tensor0StrideDotPredV1Handler() -> *mut c_void;
    fn Tensor0StrideDotS8V1Handler() -> *mut c_void;
    fn Tensor0StrideDotS16V1Handler() -> *mut c_void;
    fn Tensor0StrideDotS32V1Handler() -> *mut c_void;
    fn Tensor0StrideDotS64V1Handler() -> *mut c_void;
    fn Tensor0StrideDotU8V1Handler() -> *mut c_void;
    fn Tensor0StrideDotU16V1Handler() -> *mut c_void;
    fn Tensor0StrideDotU32V1Handler() -> *mut c_void;
    fn Tensor0StrideDotU64V1Handler() -> *mut c_void;
    fn Tensor0StrideDotF16V1Handler() -> *mut c_void;
    fn Tensor0StrideDotBF16V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideReductionS32V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionF32V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionF16V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionBF16V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionC64V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionF64V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionC128V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionS64V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionU64V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionS16V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionS8V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionU8V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionU16V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionU32V1Handler() -> *mut c_void;
    fn Tensor0StrideReductionPredV1Handler() -> *mut c_void;
    fn Tensor0StrideInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StridePreparedTypeId() -> *mut c_void;
    fn Tensor0StridePreparedTypeInfo() -> *const c_void;
    fn Tensor0StridePreparedCreatedCount() -> u64;
    fn Tensor0StridePreparedDestroyedCount() -> u64;
    fn Tensor0StrideCopyS32V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateS32V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyF32V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyF16V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateF16V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyBF16V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateBF16V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyC64V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyF64V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyC128V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateC128V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyS64V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateS64V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyU64V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateU64V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyS16V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateS16V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyS8V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateS8V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyU8V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateU8V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyU16V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateU16V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyU32V1Handler() -> *mut c_void;
    fn Tensor0StrideUpdateU32V1Handler() -> *mut c_void;
    fn Tensor0StrideCopyPredV1Handler() -> *mut c_void;
    fn Tensor0StrideUpdatePredV1Handler() -> *mut c_void;
}

#[cfg(tensor0_stride_ffi)]
fn pointer_capsule(py: Python<'_>, pointer: *mut c_void) -> PyResult<Py<PyAny>> {
    if pointer.is_null() {
        return Err(PyRuntimeError::new_err(
            "Tensor0 stride FFI returned a null pointer",
        ));
    }
    let capsule = unsafe { pyo3::ffi::PyCapsule_New(pointer, std::ptr::null(), None) };
    if capsule.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, capsule) }.unbind())
}

#[pyfunction]
pub fn _stride_ffi_available() -> bool {
    cfg!(tensor0_stride_ffi)
}

#[pyfunction]
pub fn _stride_native_registration(py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[cfg(tensor0_stride_ffi)]
    {
        let registration = PyDict::new(py);
        registration.set_item("instantiate", pointer_capsule(py, unsafe {
            Tensor0StrideInstantiateV1Handler()
        })?)?;
        registration.set_item("type_id", pointer_capsule(py, unsafe {
            Tensor0StridePreparedTypeId()
        })?)?;
        registration.set_item("type_info", pointer_capsule(py, unsafe {
            Tensor0StridePreparedTypeInfo().cast_mut()
        })?)?;
        for (name, handler) in [
            ("accumulation_instantiate", unsafe { Tensor0StrideAccumulationInstantiateV1Handler() }),
            ("accumulation_s32", unsafe { Tensor0StrideAccumulationS32V1Handler() }),
            ("accumulation_f32", unsafe { Tensor0StrideAccumulationF32V1Handler() }),
            ("accumulation_f16", unsafe { Tensor0StrideAccumulationF16V1Handler() }),
            ("accumulation_bf16", unsafe { Tensor0StrideAccumulationBF16V1Handler() }),
            ("accumulation_c64", unsafe { Tensor0StrideAccumulationC64V1Handler() }),
            ("accumulation_f64", unsafe { Tensor0StrideAccumulationF64V1Handler() }),
            ("accumulation_c128", unsafe { Tensor0StrideAccumulationC128V1Handler() }),
            ("accumulation_s64", unsafe { Tensor0StrideAccumulationS64V1Handler() }),
            ("accumulation_u64", unsafe { Tensor0StrideAccumulationU64V1Handler() }),
            ("accumulation_s16", unsafe { Tensor0StrideAccumulationS16V1Handler() }),
            ("accumulation_s8", unsafe { Tensor0StrideAccumulationS8V1Handler() }),
            ("accumulation_u8", unsafe { Tensor0StrideAccumulationU8V1Handler() }),
            ("accumulation_u16", unsafe { Tensor0StrideAccumulationU16V1Handler() }),
            ("accumulation_u32", unsafe { Tensor0StrideAccumulationU32V1Handler() }),
            ("accumulation_pred", unsafe { Tensor0StrideAccumulationPredV1Handler() }),
            ("dot_instantiate", unsafe { Tensor0StrideDotInstantiateV1Handler() }),
            ("dot_f32", unsafe { Tensor0StrideDotF32V1Handler() }),
            ("dot_pred", unsafe { Tensor0StrideDotPredV1Handler() }),
            ("dot_s8", unsafe { Tensor0StrideDotS8V1Handler() }),
            ("dot_s16", unsafe { Tensor0StrideDotS16V1Handler() }),
            ("dot_s32", unsafe { Tensor0StrideDotS32V1Handler() }),
            ("dot_s64", unsafe { Tensor0StrideDotS64V1Handler() }),
            ("dot_u8", unsafe { Tensor0StrideDotU8V1Handler() }),
            ("dot_u16", unsafe { Tensor0StrideDotU16V1Handler() }),
            ("dot_u32", unsafe { Tensor0StrideDotU32V1Handler() }),
            ("dot_u64", unsafe { Tensor0StrideDotU64V1Handler() }),
            ("dot_f16", unsafe { Tensor0StrideDotF16V1Handler() }),
            ("dot_bf16", unsafe { Tensor0StrideDotBF16V1Handler() }),
            ("dot_f64", unsafe { Tensor0StrideDotF64V1Handler() }),
            ("dot_c64", unsafe { Tensor0StrideDotC64V1Handler() }),
            ("dot_c128", unsafe { Tensor0StrideDotC128V1Handler() }),
            ("reduction_instantiate", unsafe { Tensor0StrideReductionInstantiateV1Handler() }),
            ("reduction_s32", unsafe { Tensor0StrideReductionS32V1Handler() }),
            ("reduction_f32", unsafe { Tensor0StrideReductionF32V1Handler() }),
            ("reduction_f16", unsafe { Tensor0StrideReductionF16V1Handler() }),
            ("reduction_bf16", unsafe { Tensor0StrideReductionBF16V1Handler() }),
            ("reduction_c64", unsafe { Tensor0StrideReductionC64V1Handler() }),
            ("reduction_f64", unsafe { Tensor0StrideReductionF64V1Handler() }),
            ("reduction_c128", unsafe { Tensor0StrideReductionC128V1Handler() }),
            ("reduction_s64", unsafe { Tensor0StrideReductionS64V1Handler() }),
            ("reduction_u64", unsafe { Tensor0StrideReductionU64V1Handler() }),
            ("reduction_s16", unsafe { Tensor0StrideReductionS16V1Handler() }),
            ("reduction_s8", unsafe { Tensor0StrideReductionS8V1Handler() }),
            ("reduction_u8", unsafe { Tensor0StrideReductionU8V1Handler() }),
            ("reduction_u16", unsafe { Tensor0StrideReductionU16V1Handler() }),
            ("reduction_u32", unsafe { Tensor0StrideReductionU32V1Handler() }),
            ("reduction_pred", unsafe { Tensor0StrideReductionPredV1Handler() }),
            ("copy_s32", unsafe { Tensor0StrideCopyS32V1Handler() }),
            ("update_s32", unsafe { Tensor0StrideUpdateS32V1Handler() }),
            ("copy_f32", unsafe { Tensor0StrideCopyF32V1Handler() }),
            ("update_f32", unsafe { Tensor0StrideUpdateF32V1Handler() }),
            ("copy_f16", unsafe { Tensor0StrideCopyF16V1Handler() }),
            ("update_f16", unsafe { Tensor0StrideUpdateF16V1Handler() }),
            ("copy_bf16", unsafe { Tensor0StrideCopyBF16V1Handler() }),
            ("update_bf16", unsafe { Tensor0StrideUpdateBF16V1Handler() }),
            ("copy_c64", unsafe { Tensor0StrideCopyC64V1Handler() }),
            ("update_c64", unsafe { Tensor0StrideUpdateC64V1Handler() }),
            ("copy_f64", unsafe { Tensor0StrideCopyF64V1Handler() }),
            ("update_f64", unsafe { Tensor0StrideUpdateF64V1Handler() }),
            ("copy_c128", unsafe { Tensor0StrideCopyC128V1Handler() }),
            ("update_c128", unsafe { Tensor0StrideUpdateC128V1Handler() }),
            ("copy_s64", unsafe { Tensor0StrideCopyS64V1Handler() }),
            ("update_s64", unsafe { Tensor0StrideUpdateS64V1Handler() }),
            ("copy_u64", unsafe { Tensor0StrideCopyU64V1Handler() }),
            ("update_u64", unsafe { Tensor0StrideUpdateU64V1Handler() }),
            ("copy_s16", unsafe { Tensor0StrideCopyS16V1Handler() }),
            ("update_s16", unsafe { Tensor0StrideUpdateS16V1Handler() }),
            ("copy_s8", unsafe { Tensor0StrideCopyS8V1Handler() }),
            ("update_s8", unsafe { Tensor0StrideUpdateS8V1Handler() }),
            ("copy_u8", unsafe { Tensor0StrideCopyU8V1Handler() }),
            ("update_u8", unsafe { Tensor0StrideUpdateU8V1Handler() }),
            ("copy_u16", unsafe { Tensor0StrideCopyU16V1Handler() }),
            ("update_u16", unsafe { Tensor0StrideUpdateU16V1Handler() }),
            ("copy_u32", unsafe { Tensor0StrideCopyU32V1Handler() }),
            ("update_u32", unsafe { Tensor0StrideUpdateU32V1Handler() }),
            ("copy_pred", unsafe { Tensor0StrideCopyPredV1Handler() }),
            ("update_pred", unsafe { Tensor0StrideUpdatePredV1Handler() }),
        ] {
            registration.set_item(name, pointer_capsule(py, handler)?)?;
        }
        Ok(registration.into_any().unbind())
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = py;
        Err(PyRuntimeError::new_err(
            "native CPU stride execution is unavailable",
        ))
    }
}

#[pyfunction]
pub fn _stride_native_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_ffi)]
    {
        Some(unsafe {
            (Tensor0StridePreparedCreatedCount(), Tensor0StridePreparedDestroyedCount())
        })
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        None
    }
}

#[pyfunction]
pub fn _stride_native_worker_limit() -> Option<u64> {
    #[cfg(tensor0_stride_ffi)]
    {
        let limit = unsafe { Tensor0StrideGetWorkerLimit() };
        (limit != u64::MAX).then_some(limit)
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        None
    }
}

#[pyfunction]
pub fn _set_stride_native_worker_limit(limit: Option<u64>) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetWorkerLimit(limit.unwrap_or(u64::MAX));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = limit;
    }
}

#[pyfunction]
pub fn _stride_ffi_build_versions() -> Option<(String, String)> {
    #[cfg(tensor0_stride_ffi)]
    {
        let jax = unsafe { CStr::from_ptr(Tensor0StrideBuiltJaxVersion()) };
        let jaxlib = unsafe { CStr::from_ptr(Tensor0StrideBuiltJaxlibVersion()) };
        Some((
            jax.to_string_lossy().into_owned(),
            jaxlib.to_string_lossy().into_owned(),
        ))
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        None
    }
}

pub fn add_stride_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_stride_native_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(_set_stride_native_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_registration, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_available, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_build_versions, module)?)?;
    Ok(())
}
