#[cfg(tensor0_stride_ffi)]
use std::ffi::{c_char, c_void, CStr};

use pyo3::exceptions::PyRuntimeError;
#[cfg(tensor0_stride_ffi)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

#[cfg(tensor0_stride_ffi)]
unsafe extern "C" {
    fn Tensor0StrideSumScratchCapacity(
        operation: i32,
        words: *const i64,
        word_count: usize,
        context: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void, i32, *const i64, usize, *const c_char, usize),
    );
    fn Tensor0StrideDotScratchCapacity(
        words: *const i64,
        word_count: usize,
        context: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void, i32, *const i64, usize, *const c_char, usize),
    );
    fn Tensor0StridePackOwnerFiber(
        operation: i32,
        words: *const i64,
        word_count: usize,
        context: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void, i32, *const i64, usize, *const c_char, usize),
    );
    fn Tensor0StridePrepareLayout(
        operation: i32,
        words: *const i64,
        word_count: usize,
        context: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void, i32, *const i64, usize, *const c_char, usize),
    );
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
#[derive(Default)]
struct PreparedLayoutResult {
    status: i32,
    words: Vec<i64>,
    error: Vec<u8>,
}

// The native bridge calls synchronously, borrowing valid buffers for this call
// only. Reserve fallibly before copying so no allocation panic crosses the ABI.
#[cfg(tensor0_stride_ffi)]
unsafe extern "C" fn receive_prepared_layout(
    context: *mut c_void,
    status: i32,
    words: *const i64,
    word_count: usize,
    error: *const c_char,
    error_size: usize,
) {
    let result = unsafe { &mut *context.cast::<PreparedLayoutResult>() };
    result.status = 2;
    if status == 0 {
        if result.words.try_reserve(word_count).is_err() {
            return;
        }
        if word_count != 0 {
            result
                .words
                .extend_from_slice(unsafe { std::slice::from_raw_parts(words, word_count) });
        }
    } else {
        if result.error.try_reserve(error_size).is_err() {
            return;
        }
        if error_size != 0 {
            result.error.extend_from_slice(unsafe {
                std::slice::from_raw_parts(error.cast::<u8>(), error_size)
            });
        }
    }
    result.status = status;
}

/// Fully validate and canonically sort/fuse a semantic V1 descriptor.
/// Accumulation requires provably independent output owners; Reduction
/// preserves its explicit axis roles and unsigned extent bit patterns.
/// Raises ValueError for invalid descriptors and RuntimeError for native failure.
#[pyfunction]
pub fn _stride_prepare_layout(
    py: Python<'_>,
    operation: &str,
    words: Vec<i64>,
) -> PyResult<Py<PyTuple>> {
    #[cfg(tensor0_stride_ffi)]
    {
        let operation = match operation {
            "copy" => 0,
            "update" => 1,
            "dot" => 2,
            "accumulation" => 3,
            "reduction" => 4,
            _ => return Err(PyValueError::new_err("unsupported stride layout operation")),
        };
        let mut result = PreparedLayoutResult {
            status: 2,
            ..Default::default()
        };
        // No pointers escape this synchronous call, including the stack context.
        unsafe {
            Tensor0StridePrepareLayout(
                operation,
                words.as_ptr(),
                words.len(),
                (&mut result as *mut PreparedLayoutResult).cast(),
                receive_prepared_layout,
            );
        }
        if result.status != 0 {
            let message = if result.error.is_empty() {
                "native stride layout preparation failed".into()
            } else {
                String::from_utf8_lossy(&result.error).into_owned()
            };
            return Err(if result.status == 1 {
                PyValueError::new_err(message)
            } else {
                PyRuntimeError::new_err(message)
            });
        }
        Ok(PyTuple::new(py, result.words)?.unbind())
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = (py, operation, words);
        Err(PyRuntimeError::new_err(
            "native stride layout preparation is unavailable",
        ))
    }
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

#[cfg(tensor0_stride_cuda)]
unsafe extern "C" {
    fn Tensor0StrideCudaCopyInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideCudaCopyPreparedTypeId() -> *mut c_void;
    fn Tensor0StrideCudaCopyPreparedTypeInfo() -> *const c_void;
    fn Tensor0StrideCudaCopyPreparedCreatedCount() -> u64;
    fn Tensor0StrideCudaCopyPreparedDestroyedCount() -> u64;
    fn Tensor0StrideCudaCopyF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaCopyF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaCopyC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaCopyC128V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaUpdateInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideCudaUpdatePreparedTypeId() -> *mut c_void;
    fn Tensor0StrideCudaUpdatePreparedTypeInfo() -> *const c_void;
    fn Tensor0StrideCudaUpdatePreparedCreatedCount() -> u64;
    fn Tensor0StrideCudaUpdatePreparedDestroyedCount() -> u64;
    fn Tensor0StrideCudaUpdateF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaUpdateF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaUpdateC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaUpdateC128V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaDotInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideCudaDotPreparedTypeId() -> *mut c_void;
    fn Tensor0StrideCudaDotPreparedTypeInfo() -> *const c_void;
    fn Tensor0StrideCudaDotPreparedCreatedCount() -> u64;
    fn Tensor0StrideCudaDotPreparedDestroyedCount() -> u64;
    fn Tensor0StrideCudaDotF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaDotF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaDotC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaDotC128V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationPreparedTypeId() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationPreparedTypeInfo() -> *const c_void;
    fn Tensor0StrideCudaAccumulationPreparedCreatedCount() -> u64;
    fn Tensor0StrideCudaAccumulationPreparedDestroyedCount() -> u64;
    fn Tensor0StrideCudaAccumulationF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaReductionInstantiateV1Handler() -> *mut c_void;
    fn Tensor0StrideCudaReductionPreparedTypeId() -> *mut c_void;
    fn Tensor0StrideCudaReductionPreparedTypeInfo() -> *const c_void;
    fn Tensor0StrideCudaReductionPreparedCreatedCount() -> u64;
    fn Tensor0StrideCudaReductionPreparedDestroyedCount() -> u64;
    fn Tensor0StrideCudaReductionF32V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaReductionF64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaReductionC64V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaAccumulationC128V1Handler() -> *mut c_void;
    fn Tensor0StrideCudaReductionC128V1Handler() -> *mut c_void;

}

#[pyfunction]
pub fn _stride_cuda_available() -> bool {
    cfg!(tensor0_stride_cuda)
}

#[pyfunction]
pub fn _stride_cuda_registration(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let registration = PyDict::new(py);
    #[cfg(tensor0_stride_cuda)]
    for (name, handler) in [
        ("copy_instantiate", unsafe { Tensor0StrideCudaCopyInstantiateV1Handler() }),
        ("copy_type_id", unsafe { Tensor0StrideCudaCopyPreparedTypeId() }),
        ("copy_type_info", unsafe { Tensor0StrideCudaCopyPreparedTypeInfo().cast_mut() }),
        ("copy_f32", unsafe { Tensor0StrideCudaCopyF32V1Handler() }),
        ("copy_f64", unsafe { Tensor0StrideCudaCopyF64V1Handler() }),
        ("copy_c64", unsafe { Tensor0StrideCudaCopyC64V1Handler() }),
        ("copy_c128", unsafe { Tensor0StrideCudaCopyC128V1Handler() }),
        ("update_instantiate", unsafe { Tensor0StrideCudaUpdateInstantiateV1Handler() }),
        ("update_type_id", unsafe { Tensor0StrideCudaUpdatePreparedTypeId() }),
        ("update_type_info", unsafe { Tensor0StrideCudaUpdatePreparedTypeInfo().cast_mut() }),
        ("update_f32", unsafe { Tensor0StrideCudaUpdateF32V1Handler() }),
        ("update_f64", unsafe { Tensor0StrideCudaUpdateF64V1Handler() }),
        ("update_c64", unsafe { Tensor0StrideCudaUpdateC64V1Handler() }),
        ("update_c128", unsafe { Tensor0StrideCudaUpdateC128V1Handler() }),
        ("dot_instantiate", unsafe { Tensor0StrideCudaDotInstantiateV1Handler() }),
        ("dot_type_id", unsafe { Tensor0StrideCudaDotPreparedTypeId() }),
        ("dot_type_info", unsafe { Tensor0StrideCudaDotPreparedTypeInfo().cast_mut() }),
        ("dot_f32", unsafe { Tensor0StrideCudaDotF32V1Handler() }),
        ("dot_f64", unsafe { Tensor0StrideCudaDotF64V1Handler() }),
        ("dot_c64", unsafe { Tensor0StrideCudaDotC64V1Handler() }),
        ("dot_c128", unsafe { Tensor0StrideCudaDotC128V1Handler() }),
        ("accumulation_instantiate", unsafe { Tensor0StrideCudaAccumulationInstantiateV1Handler() }),
        ("accumulation_type_id", unsafe { Tensor0StrideCudaAccumulationPreparedTypeId() }),
        ("accumulation_type_info", unsafe { Tensor0StrideCudaAccumulationPreparedTypeInfo().cast_mut() }),
        ("accumulation_f32", unsafe { Tensor0StrideCudaAccumulationF32V1Handler() }),
        ("reduction_instantiate", unsafe { Tensor0StrideCudaReductionInstantiateV1Handler() }),
        ("reduction_type_id", unsafe { Tensor0StrideCudaReductionPreparedTypeId() }),
        ("reduction_type_info", unsafe { Tensor0StrideCudaReductionPreparedTypeInfo().cast_mut() }),
        ("reduction_f32", unsafe { Tensor0StrideCudaReductionF32V1Handler() }),
        ("accumulation_f64", unsafe { Tensor0StrideCudaAccumulationF64V1Handler() }),
        ("reduction_f64", unsafe { Tensor0StrideCudaReductionF64V1Handler() }),
        ("accumulation_c64", unsafe { Tensor0StrideCudaAccumulationC64V1Handler() }),
        ("reduction_c64", unsafe { Tensor0StrideCudaReductionC64V1Handler() }),
        ("accumulation_c128", unsafe { Tensor0StrideCudaAccumulationC128V1Handler() }),
        ("reduction_c128", unsafe { Tensor0StrideCudaReductionC128V1Handler() }),

    ] {
        registration.set_item(name, pointer_capsule(py, handler)?)?;
    }
    Ok(registration.into_any().unbind())
}

#[pyfunction]
pub fn _stride_cuda_copy_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_cuda)]
    {
        Some(unsafe {
            (
                Tensor0StrideCudaCopyPreparedCreatedCount(),
                Tensor0StrideCudaCopyPreparedDestroyedCount(),
            )
        })
    }
    #[cfg(not(tensor0_stride_cuda))]
    {
        None
    }
}

#[pyfunction]
pub fn _stride_cuda_update_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_cuda)]
    {
        Some(unsafe {
            (
                Tensor0StrideCudaUpdatePreparedCreatedCount(),
                Tensor0StrideCudaUpdatePreparedDestroyedCount(),
            )
        })
    }
    #[cfg(not(tensor0_stride_cuda))]
    {
        None
    }
}

#[pyfunction]
pub fn _stride_cuda_dot_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_cuda)]
    {
        Some(unsafe {
            (
                Tensor0StrideCudaDotPreparedCreatedCount(),
                Tensor0StrideCudaDotPreparedDestroyedCount(),
            )
        })
    }
    #[cfg(not(tensor0_stride_cuda))]
    {
        None
    }
}

#[pyfunction]
pub fn _stride_cuda_accumulation_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_cuda)]
    {
        Some(unsafe {
            (
                Tensor0StrideCudaAccumulationPreparedCreatedCount(),
                Tensor0StrideCudaAccumulationPreparedDestroyedCount(),
            )
        })
    }
    #[cfg(not(tensor0_stride_cuda))]
    {
        None
    }
}

#[pyfunction]
pub fn _stride_cuda_reduction_prepared_stats() -> Option<(u64, u64)> {
    #[cfg(tensor0_stride_cuda)]
    {
        Some(unsafe {
            (
                Tensor0StrideCudaReductionPreparedCreatedCount(),
                Tensor0StrideCudaReductionPreparedDestroyedCount(),
            )
        })
    }
    #[cfg(not(tensor0_stride_cuda))]
    {
        None
    }
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

/// Encode validated records as a CUDA owner/fiber execution view.
#[pyfunction]
pub fn _stride_pack_owner_fiber(
    py: Python<'_>, operation: &str, words: Vec<i64>,
) -> PyResult<Py<PyTuple>> {
    #[cfg(tensor0_stride_ffi)]
    {
        let operation = match operation {
            "copy" => 0,
            "update" => 1,
            "dot" => 2,
            "accumulation" => 3,
            "reduction" => 4,
            _ => return Err(PyValueError::new_err("unsupported owner/fiber operation")),
        };
        let mut result = PreparedLayoutResult { status: 2, ..Default::default() };
        unsafe {
            Tensor0StridePackOwnerFiber(operation, words.as_ptr(), words.len(),
                (&mut result as *mut PreparedLayoutResult).cast(), receive_prepared_layout);
        }
        if result.status != 0 {
            let message = if result.error.is_empty() {
                "native owner/fiber encoding failed".into()
            } else {
                String::from_utf8_lossy(&result.error).into_owned()
            };
            return Err(if result.status == 1 {
                PyValueError::new_err(message)
            } else {
                PyRuntimeError::new_err(message)
            });
        }
        Ok(PyTuple::new(py, result.words)?.unbind())
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = (py, operation, words);
        Err(PyRuntimeError::new_err("native stride owner/fiber encoding is unavailable"))
    }
}

/// Scratch slots per batch for prepared Accumulation or Reduction.
#[pyfunction]
pub fn _stride_sum_scratch_capacity(operation: &str, words: Vec<i64>) -> PyResult<i64> {
    #[cfg(tensor0_stride_ffi)]
    {
        let operation = match operation {
            "accumulation" => 3,
            "reduction" => 4,
            _ => return Err(PyValueError::new_err("unsupported sum scratch operation")),
        };
        let mut result = PreparedLayoutResult { status: 2, ..Default::default() };
        unsafe {
            Tensor0StrideSumScratchCapacity(operation, words.as_ptr(), words.len(),
                (&mut result as *mut PreparedLayoutResult).cast(), receive_prepared_layout);
        }
        if result.status != 0 {
            let message = if result.error.is_empty() {
                "native sum scratch preparation failed".into()
            } else {
                String::from_utf8_lossy(&result.error).into_owned()
            };
            return Err(if result.status == 1 {
                PyValueError::new_err(message)
            } else {
                PyRuntimeError::new_err(message)
            });
        }
        if result.words.len() != 1 {
            return Err(PyRuntimeError::new_err("native sum scratch preparation returned invalid capacity"));
        }
        Ok(result.words[0])
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = (operation, words);
        Err(PyRuntimeError::new_err("native sum scratch preparation is unavailable"))
    }
}

/// Scratch slots per batch for a prepared Dot descriptor.
#[pyfunction]
pub fn _stride_dot_scratch_capacity(words: Vec<i64>) -> PyResult<i64> {
    #[cfg(tensor0_stride_ffi)]
    {
        let mut result = PreparedLayoutResult { status: 2, ..Default::default() };
        unsafe {
            Tensor0StrideDotScratchCapacity(words.as_ptr(), words.len(),
                (&mut result as *mut PreparedLayoutResult).cast(), receive_prepared_layout);
        }
        if result.status != 0 {
            let message = if result.error.is_empty() {
                "native dot scratch preparation failed".into()
            } else {
                String::from_utf8_lossy(&result.error).into_owned()
            };
            return Err(if result.status == 1 {
                PyValueError::new_err(message)
            } else {
                PyRuntimeError::new_err(message)
            });
        }
        if result.words.len() != 1 {
            return Err(PyRuntimeError::new_err("native dot scratch preparation returned invalid capacity"));
        }
        Ok(result.words[0])
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = words;
        Err(PyRuntimeError::new_err("native dot scratch preparation is unavailable"))
    }
}

pub fn add_stride_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_stride_prepare_layout, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_dot_scratch_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_sum_scratch_capacity, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_pack_owner_fiber, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_registration, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_copy_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_update_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_accumulation_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_dot_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_cuda_reduction_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(_set_stride_native_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_registration, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_prepared_stats, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_available, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_build_versions, module)?)?;
    Ok(())
}
