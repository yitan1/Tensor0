#[cfg(tensor0_stride_ffi)]
use std::ffi::{c_char, c_void, CStr};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[cfg(tensor0_stride_ffi)]
unsafe extern "C" {
    fn Tensor0StrideR2PreparedF32V7InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedF32V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedF16V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedBF16V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedC64V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedS32V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedPredV7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedS8V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedS16V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedS64V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedU8V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedU16V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedU32V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedU64V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedF64V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideR2PreparedC128V7ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF16F32ForwardV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF32F16TransposeV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF32C64ForwardV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineC64F32TransposeV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineC64F32ForwardV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF32C64TransposeV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF64C128ForwardV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineC128F64TransposeV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineC128F64ForwardV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffineF64C128TransposeV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignPredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignU8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignU16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignU32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignU64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF16F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF32C64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignC64F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignF64C128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignC128F64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS32PredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS32S8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignS32S16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAssignU32U8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulatePredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateU8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateU16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateU32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateU64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF16F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF32C64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateC64F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateF64C128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateC128F64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS32PredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS32S8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateS32S16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideBaseAccumulateU32U8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleS32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScalePredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleS8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleS16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleS64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleU8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleU16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleU32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleU64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF32F16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC64F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF32C64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC128F64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF64C128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF32ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF16ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionBF16ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionPredForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionS8ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionS16ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionS32ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionS64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionU8ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionU16ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionU32ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionU64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF16F32ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF32C64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC64F32ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC128ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionF64C128ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionC128F64ForwardV2ExecuteHandler() -> *mut c_void;
    fn Tensor0StridePreparedTypeId() -> *mut c_void;
    fn Tensor0StridePreparedTypeInfo() -> *const c_void;
    fn Tensor0StridePreparedInstantiateCount() -> u64;
    fn Tensor0StridePreparedExecuteCount() -> u64;
    fn Tensor0StridePreparedLiveStateCount() -> u64;
    fn Tensor0StridePreparedDestroyedStateCount() -> u64;
    fn Tensor0StridePreparedLiveBytes() -> u64;
    fn Tensor0StridePreparedLastStateBytes() -> u64;
    fn Tensor0StridePreparedLastDescriptorBytes() -> u64;
    fn Tensor0StridePreparedResetMetrics() -> u64;
    fn Tensor0StrideNativeCallCount() -> u64;
    fn Tensor0StrideResetNativeCallCount();
    fn Tensor0StrideLastWorkerCount() -> u64;
    fn Tensor0StrideLastAvailableWorkerCount() -> u64;
    fn Tensor0StrideSetWorkerLimitForTests(limit: u64);
    fn Tensor0StrideBuiltJaxVersion() -> *const c_char;
    fn Tensor0StrideBuiltJaxlibVersion() -> *const c_char;
    fn Tensor0StrideAbiVersion() -> u64;
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
pub fn _stride_prepared_registration(py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[cfg(tensor0_stride_ffi)]
    {
        let registration = PyDict::new(py);
        registration.set_item(
            "instantiate",
            pointer_capsule(py, unsafe {
                Tensor0StrideR2PreparedF32V7InstantiateHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f32",
            pointer_capsule(py, unsafe { Tensor0StrideR2PreparedF32V7ExecuteHandler() })?,
        )?;
        registration.set_item(
            "execute_f16",
            pointer_capsule(py, unsafe { Tensor0StrideR2PreparedF16V7ExecuteHandler() })?,
        )?;
        registration.set_item(
            "execute_bf16",
            pointer_capsule(py, unsafe { Tensor0StrideR2PreparedBF16V7ExecuteHandler() })?,
        )?;
        registration.set_item(
            "execute_c64",
            pointer_capsule(py, unsafe { Tensor0StrideR2PreparedC64V7ExecuteHandler() })?,
        )?;
        registration.set_item(
            "execute_s32",
            pointer_capsule(py, unsafe { Tensor0StrideR2PreparedS32V7ExecuteHandler() })?,
        )?;
        let same_dtype_handlers = unsafe {
            [
                (
                    "execute_pred",
                    Tensor0StrideR2PreparedPredV7ExecuteHandler(),
                ),
                ("execute_s8", Tensor0StrideR2PreparedS8V7ExecuteHandler()),
                ("execute_s16", Tensor0StrideR2PreparedS16V7ExecuteHandler()),
                ("execute_s64", Tensor0StrideR2PreparedS64V7ExecuteHandler()),
                ("execute_u8", Tensor0StrideR2PreparedU8V7ExecuteHandler()),
                ("execute_u16", Tensor0StrideR2PreparedU16V7ExecuteHandler()),
                ("execute_u32", Tensor0StrideR2PreparedU32V7ExecuteHandler()),
                ("execute_u64", Tensor0StrideR2PreparedU64V7ExecuteHandler()),
                ("execute_f64", Tensor0StrideR2PreparedF64V7ExecuteHandler()),
                (
                    "execute_c128",
                    Tensor0StrideR2PreparedC128V7ExecuteHandler(),
                ),
            ]
        };
        for (name, handler) in same_dtype_handlers {
            registration.set_item(name, pointer_capsule(py, handler)?)?;
        }
        registration.set_item(
            "execute_f16_f32_forward",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineF16F32ForwardV1ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f32_f16_transpose",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineF32F16TransposeV1ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f32_c64_forward",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineF32C64ForwardV1ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_c64_f32_transpose",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineC64F32TransposeV1ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_c64_f32_forward",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineC64F32ForwardV1ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f32_c64_transpose",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffineF32C64TransposeV1ExecuteHandler()
            })?,
        )?;
        let wide_mixed_handlers = unsafe {
            [
                (
                    "execute_f64_c128_forward",
                    Tensor0StrideAffineF64C128ForwardV1ExecuteHandler(),
                ),
                (
                    "execute_c128_f64_transpose",
                    Tensor0StrideAffineC128F64TransposeV1ExecuteHandler(),
                ),
                (
                    "execute_c128_f64_forward",
                    Tensor0StrideAffineC128F64ForwardV1ExecuteHandler(),
                ),
                (
                    "execute_f64_c128_transpose",
                    Tensor0StrideAffineF64C128TransposeV1ExecuteHandler(),
                ),
            ]
        };
        for (name, handler) in wide_mixed_handlers {
            registration.set_item(name, pointer_capsule(py, handler)?)?;
        }
        let operation_handlers = unsafe {
            [
                (
                    "instantiate_base_assign",
                    Tensor0StrideBaseAssignV1InstantiateHandler(),
                ),
                (
                    "instantiate_base_accumulate",
                    Tensor0StrideBaseAccumulateV1InstantiateHandler(),
                ),
                (
                    "instantiate_selected_scale",
                    Tensor0StrideSelectedScaleV1InstantiateHandler(),
                ),
                (
                    "execute_base_assign_f32",
                    Tensor0StrideBaseAssignF32V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_f16",
                    Tensor0StrideBaseAssignF16V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_bf16",
                    Tensor0StrideBaseAssignBF16V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_c64",
                    Tensor0StrideBaseAssignC64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s32",
                    Tensor0StrideBaseAssignS32V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_pred",
                    Tensor0StrideBaseAssignPredV1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s8",
                    Tensor0StrideBaseAssignS8V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s16",
                    Tensor0StrideBaseAssignS16V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s64",
                    Tensor0StrideBaseAssignS64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_u8",
                    Tensor0StrideBaseAssignU8V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_u16",
                    Tensor0StrideBaseAssignU16V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_u32",
                    Tensor0StrideBaseAssignU32V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_u64",
                    Tensor0StrideBaseAssignU64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_f64",
                    Tensor0StrideBaseAssignF64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_c128",
                    Tensor0StrideBaseAssignC128V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_f16_f32",
                    Tensor0StrideBaseAssignF16F32V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_f32_c64",
                    Tensor0StrideBaseAssignF32C64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_c64_f32",
                    Tensor0StrideBaseAssignC64F32V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_f64_c128",
                    Tensor0StrideBaseAssignF64C128V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_c128_f64",
                    Tensor0StrideBaseAssignC128F64V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s32_pred",
                    Tensor0StrideBaseAssignS32PredV1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s32_s8",
                    Tensor0StrideBaseAssignS32S8V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_s32_s16",
                    Tensor0StrideBaseAssignS32S16V1ExecuteHandler(),
                ),
                (
                    "execute_base_assign_u32_u8",
                    Tensor0StrideBaseAssignU32U8V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f32",
                    Tensor0StrideBaseAccumulateF32V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f16",
                    Tensor0StrideBaseAccumulateF16V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_bf16",
                    Tensor0StrideBaseAccumulateBF16V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_c64",
                    Tensor0StrideBaseAccumulateC64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s32",
                    Tensor0StrideBaseAccumulateS32V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_pred",
                    Tensor0StrideBaseAccumulatePredV1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s8",
                    Tensor0StrideBaseAccumulateS8V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s16",
                    Tensor0StrideBaseAccumulateS16V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s64",
                    Tensor0StrideBaseAccumulateS64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_u8",
                    Tensor0StrideBaseAccumulateU8V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_u16",
                    Tensor0StrideBaseAccumulateU16V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_u32",
                    Tensor0StrideBaseAccumulateU32V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_u64",
                    Tensor0StrideBaseAccumulateU64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f64",
                    Tensor0StrideBaseAccumulateF64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_c128",
                    Tensor0StrideBaseAccumulateC128V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f16_f32",
                    Tensor0StrideBaseAccumulateF16F32V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f32_c64",
                    Tensor0StrideBaseAccumulateF32C64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_c64_f32",
                    Tensor0StrideBaseAccumulateC64F32V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_f64_c128",
                    Tensor0StrideBaseAccumulateF64C128V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_c128_f64",
                    Tensor0StrideBaseAccumulateC128F64V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s32_pred",
                    Tensor0StrideBaseAccumulateS32PredV1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s32_s8",
                    Tensor0StrideBaseAccumulateS32S8V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_s32_s16",
                    Tensor0StrideBaseAccumulateS32S16V1ExecuteHandler(),
                ),
                (
                    "execute_base_accumulate_u32_u8",
                    Tensor0StrideBaseAccumulateU32U8V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_f32",
                    Tensor0StrideSelectedScaleF32V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_f16",
                    Tensor0StrideSelectedScaleF16V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_bf16",
                    Tensor0StrideSelectedScaleBF16V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_c64",
                    Tensor0StrideSelectedScaleC64V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_s32",
                    Tensor0StrideSelectedScaleS32V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_pred",
                    Tensor0StrideSelectedScalePredV1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_s8",
                    Tensor0StrideSelectedScaleS8V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_s16",
                    Tensor0StrideSelectedScaleS16V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_s64",
                    Tensor0StrideSelectedScaleS64V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_u8",
                    Tensor0StrideSelectedScaleU8V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_u16",
                    Tensor0StrideSelectedScaleU16V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_u32",
                    Tensor0StrideSelectedScaleU32V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_u64",
                    Tensor0StrideSelectedScaleU64V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_f64",
                    Tensor0StrideSelectedScaleF64V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_c128",
                    Tensor0StrideSelectedScaleC128V1ExecuteHandler(),
                ),
                (
                    "instantiate_structured_reduction",
                    Tensor0StrideStructuredReductionV1InstantiateHandler(),
                ),
                (
                    "execute_structured_reduction_f32",
                    Tensor0StrideStructuredReductionF32V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_f16",
                    Tensor0StrideStructuredReductionF16V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_bf16",
                    Tensor0StrideStructuredReductionBF16V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_c64",
                    Tensor0StrideStructuredReductionC64V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_f32_f16",
                    Tensor0StrideStructuredReductionF32F16V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_c64_f32",
                    Tensor0StrideStructuredReductionC64F32V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_f32_c64",
                    Tensor0StrideStructuredReductionF32C64V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_f64",
                    Tensor0StrideStructuredReductionF64V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_c128",
                    Tensor0StrideStructuredReductionC128V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_c128_f64",
                    Tensor0StrideStructuredReductionC128F64V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_f64_c128",
                    Tensor0StrideStructuredReductionF64C128V1ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f32",
                    Tensor0StrideStructuredReductionF32ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f16",
                    Tensor0StrideStructuredReductionF16ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_bf16",
                    Tensor0StrideStructuredReductionBF16ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_c64",
                    Tensor0StrideStructuredReductionC64ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_pred",
                    Tensor0StrideStructuredReductionPredForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_s8",
                    Tensor0StrideStructuredReductionS8ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_s16",
                    Tensor0StrideStructuredReductionS16ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_s32",
                    Tensor0StrideStructuredReductionS32ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_s64",
                    Tensor0StrideStructuredReductionS64ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_u8",
                    Tensor0StrideStructuredReductionU8ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_u16",
                    Tensor0StrideStructuredReductionU16ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_u32",
                    Tensor0StrideStructuredReductionU32ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_u64",
                    Tensor0StrideStructuredReductionU64ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f16_f32",
                    Tensor0StrideStructuredReductionF16F32ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f32_c64",
                    Tensor0StrideStructuredReductionF32C64ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_c64_f32",
                    Tensor0StrideStructuredReductionC64F32ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f64",
                    Tensor0StrideStructuredReductionF64ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_c128",
                    Tensor0StrideStructuredReductionC128ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_f64_c128",
                    Tensor0StrideStructuredReductionF64C128ForwardV2ExecuteHandler(),
                ),
                (
                    "execute_structured_reduction_forward_c128_f64",
                    Tensor0StrideStructuredReductionC128F64ForwardV2ExecuteHandler(),
                ),
            ]
        };
        for (name, handler) in operation_handlers {
            registration.set_item(name, pointer_capsule(py, handler)?)?;
        }
        registration.set_item(
            "type_id",
            pointer_capsule(py, unsafe { Tensor0StridePreparedTypeId() })?,
        )?;
        registration.set_item(
            "type_info",
            pointer_capsule(py, unsafe { Tensor0StridePreparedTypeInfo().cast_mut() })?,
        )?;
        Ok(registration.into_any().unbind())
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = py;
        Err(PyRuntimeError::new_err(
            "Tensor0 was built without JAX FFI headers",
        ))
    }
}

#[pyfunction]
pub fn _stride_prepared_metrics() -> (u64, u64, u64, u64, u64, u64, u64) {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe {
            (
                Tensor0StridePreparedInstantiateCount(),
                Tensor0StridePreparedExecuteCount(),
                Tensor0StridePreparedLiveStateCount(),
                Tensor0StridePreparedDestroyedStateCount(),
                Tensor0StridePreparedLiveBytes(),
                Tensor0StridePreparedLastStateBytes(),
                Tensor0StridePreparedLastDescriptorBytes(),
            )
        }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        (0, 0, 0, 0, 0, 0, 0)
    }
}

#[pyfunction]
pub fn _stride_prepared_reset_metrics() -> bool {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StridePreparedResetMetrics() != 0 }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        false
    }
}

#[pyfunction]
pub fn _stride_native_call_count() -> u64 {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StrideNativeCallCount() }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        0
    }
}

#[pyfunction]
pub fn _reset_stride_native_call_count() {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideResetNativeCallCount();
    }
}

#[pyfunction]
pub fn _stride_last_worker_count() -> u64 {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StrideLastWorkerCount() }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        0
    }
}

#[pyfunction]
pub fn _stride_last_available_worker_count() -> u64 {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StrideLastAvailableWorkerCount() }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        0
    }
}

#[pyfunction]
pub fn _set_stride_worker_limit_for_tests(limit: Option<u64>) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetWorkerLimitForTests(limit.unwrap_or(u64::MAX));
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

#[pyfunction]
pub fn _stride_ffi_abi_version() -> Option<u64> {
    #[cfg(tensor0_stride_ffi)]
    {
        Some(unsafe { Tensor0StrideAbiVersion() })
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        None
    }
}

pub fn add_stride_functions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_stride_ffi_available, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_prepared_registration, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_prepared_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_prepared_reset_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_native_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(_reset_stride_native_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_last_worker_count, module)?)?;
    module.add_function(wrap_pyfunction!(
        _stride_last_available_worker_count,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_worker_limit_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_build_versions, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_abi_version, module)?)?;
    Ok(())
}
