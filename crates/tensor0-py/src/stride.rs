#[cfg(tensor0_stride_ffi)]
use std::ffi::{c_char, c_void, CStr};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[cfg(tensor0_stride_ffi)]
unsafe extern "C" {
    fn Tensor0StrideAffinePreparedF32V6InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedF32V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedF16V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedBF16V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedC64V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedS32V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedPredV6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedS8V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedS16V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedS64V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedU8V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedU16V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedU32V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedU64V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedF64V6ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAffinePreparedC128V6ExecuteHandler() -> *mut c_void;
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
    fn Tensor0StrideSelectedScaleV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideUpdateV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdatePredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateU8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateU16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateU32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateU64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF16F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF32C64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateC64F32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateF64C128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateC128F64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS32PredV1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS32S8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateS32S16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideUpdateU32U8V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideAxpbyC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideDotV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideDotuF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideDotcF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideDotuC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideDotcC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentBF16V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentF64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideScaleTangentC128V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleAliasF32V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideSelectedScaleAliasC64V1ExecuteHandler() -> *mut c_void;
    fn Tensor0StrideStructuredReductionV1InstantiateHandler() -> *mut c_void;
    fn Tensor0StrideGroupedReductionV1InstantiateHandler() -> *mut c_void;
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
    fn Tensor0StrideObserveLeafKernelsForTests(enabled: u64);
    fn Tensor0StrideLeafKernelMaskForTests() -> u64;
    fn Tensor0StrideSupportedLeafKernelMaskForTests() -> u64;
    fn Tensor0StrideResetNativeCallCount();
    fn Tensor0StrideLastWorkerCount() -> u64;
    fn Tensor0StrideLastAvailableWorkerCount() -> u64;
    fn Tensor0StrideLastReductionFiberChunks() -> u64;
    fn Tensor0StrideLastGroupedOutputOwner() -> u64;
    fn Tensor0StrideAliasLastBasePointer() -> u64;
    fn Tensor0StrideAliasLastSourcePointer() -> u64;
    fn Tensor0StrideAliasLastFactorPointer() -> u64;
    fn Tensor0StrideAliasLastResultPointer() -> u64;
    fn Tensor0StrideResetAliasPointers();
    fn Tensor0StrideGetWorkerLimit() -> u64;
    fn Tensor0StrideSetWorkerLimit(limit: u64);
    fn Tensor0StrideSetForceGenericForTests(enabled: u64);
    fn Tensor0StrideSetForceGeneratedBaselineForTests(enabled: u64);
    fn Tensor0StrideSetReductionFiberParallelModeForTests(mode: u64);
    fn Tensor0StrideSetDisableF16F16ContiguousSimdForTests(enabled: u64);
    fn Tensor0StrideSetDisableF16F32ContiguousSimdForTests(enabled: u64);
    fn Tensor0StrideSetDisableF32C64ContiguousSimdForTests(enabled: u64);
    fn Tensor0StrideCpuSupportsF16ContiguousSimd() -> u64;
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
                Tensor0StrideAffinePreparedF32V6InstantiateHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f32",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffinePreparedF32V6ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_f16",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffinePreparedF16V6ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_bf16",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffinePreparedBF16V6ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_c64",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffinePreparedC64V6ExecuteHandler()
            })?,
        )?;
        registration.set_item(
            "execute_s32",
            pointer_capsule(py, unsafe {
                Tensor0StrideAffinePreparedS32V6ExecuteHandler()
            })?,
        )?;
        let same_dtype_handlers = unsafe {
            [
                (
                    "execute_pred",
                    Tensor0StrideAffinePreparedPredV6ExecuteHandler(),
                ),
                (
                    "execute_s8",
                    Tensor0StrideAffinePreparedS8V6ExecuteHandler(),
                ),
                (
                    "execute_s16",
                    Tensor0StrideAffinePreparedS16V6ExecuteHandler(),
                ),
                (
                    "execute_s64",
                    Tensor0StrideAffinePreparedS64V6ExecuteHandler(),
                ),
                (
                    "execute_u8",
                    Tensor0StrideAffinePreparedU8V6ExecuteHandler(),
                ),
                (
                    "execute_u16",
                    Tensor0StrideAffinePreparedU16V6ExecuteHandler(),
                ),
                (
                    "execute_u32",
                    Tensor0StrideAffinePreparedU32V6ExecuteHandler(),
                ),
                (
                    "execute_u64",
                    Tensor0StrideAffinePreparedU64V6ExecuteHandler(),
                ),
                (
                    "execute_f64",
                    Tensor0StrideAffinePreparedF64V6ExecuteHandler(),
                ),
                (
                    "execute_c128",
                    Tensor0StrideAffinePreparedC128V6ExecuteHandler(),
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
                    "instantiate_selected_scale",
                    Tensor0StrideSelectedScaleV1InstantiateHandler(),
                ),
                (
                    "instantiate_update",
                    Tensor0StrideUpdateV1InstantiateHandler(),
                ),
                ("instantiate_dot", Tensor0StrideDotV1InstantiateHandler()),
                (
                    "execute_scale_tangent_f16",
                    Tensor0StrideScaleTangentF16V1ExecuteHandler(),
                ),
                (
                    "execute_scale_tangent_bf16",
                    Tensor0StrideScaleTangentBF16V1ExecuteHandler(),
                ),
                (
                    "execute_scale_tangent_f32",
                    Tensor0StrideScaleTangentF32V1ExecuteHandler(),
                ),
                (
                    "execute_scale_tangent_f64",
                    Tensor0StrideScaleTangentF64V1ExecuteHandler(),
                ),
                (
                    "execute_scale_tangent_c64",
                    Tensor0StrideScaleTangentC64V1ExecuteHandler(),
                ),
                (
                    "execute_scale_tangent_c128",
                    Tensor0StrideScaleTangentC128V1ExecuteHandler(),
                ),
                ("execute_axpby_f16", Tensor0StrideAxpbyF16V1ExecuteHandler()),
                (
                    "execute_update_f32",
                    Tensor0StrideUpdateF32V1ExecuteHandler(),
                ),
                (
                    "execute_update_f16",
                    Tensor0StrideUpdateF16V1ExecuteHandler(),
                ),
                (
                    "execute_update_bf16",
                    Tensor0StrideUpdateBF16V1ExecuteHandler(),
                ),
                (
                    "execute_update_c64",
                    Tensor0StrideUpdateC64V1ExecuteHandler(),
                ),
                (
                    "execute_update_s32",
                    Tensor0StrideUpdateS32V1ExecuteHandler(),
                ),
                (
                    "execute_update_pred",
                    Tensor0StrideUpdatePredV1ExecuteHandler(),
                ),
                ("execute_update_s8", Tensor0StrideUpdateS8V1ExecuteHandler()),
                (
                    "execute_update_s16",
                    Tensor0StrideUpdateS16V1ExecuteHandler(),
                ),
                (
                    "execute_update_s64",
                    Tensor0StrideUpdateS64V1ExecuteHandler(),
                ),
                ("execute_update_u8", Tensor0StrideUpdateU8V1ExecuteHandler()),
                (
                    "execute_update_u16",
                    Tensor0StrideUpdateU16V1ExecuteHandler(),
                ),
                (
                    "execute_update_u32",
                    Tensor0StrideUpdateU32V1ExecuteHandler(),
                ),
                (
                    "execute_update_u64",
                    Tensor0StrideUpdateU64V1ExecuteHandler(),
                ),
                (
                    "execute_update_f64",
                    Tensor0StrideUpdateF64V1ExecuteHandler(),
                ),
                (
                    "execute_update_c128",
                    Tensor0StrideUpdateC128V1ExecuteHandler(),
                ),
                (
                    "execute_update_f16_f32",
                    Tensor0StrideUpdateF16F32V1ExecuteHandler(),
                ),
                (
                    "execute_update_f32_c64",
                    Tensor0StrideUpdateF32C64V1ExecuteHandler(),
                ),
                (
                    "execute_update_c64_f32",
                    Tensor0StrideUpdateC64F32V1ExecuteHandler(),
                ),
                (
                    "execute_update_f64_c128",
                    Tensor0StrideUpdateF64C128V1ExecuteHandler(),
                ),
                (
                    "execute_update_c128_f64",
                    Tensor0StrideUpdateC128F64V1ExecuteHandler(),
                ),
                (
                    "execute_update_s32_pred",
                    Tensor0StrideUpdateS32PredV1ExecuteHandler(),
                ),
                (
                    "execute_update_s32_s8",
                    Tensor0StrideUpdateS32S8V1ExecuteHandler(),
                ),
                (
                    "execute_update_s32_s16",
                    Tensor0StrideUpdateS32S16V1ExecuteHandler(),
                ),
                (
                    "execute_update_u32_u8",
                    Tensor0StrideUpdateU32U8V1ExecuteHandler(),
                ),
                (
                    "execute_axpby_bf16",
                    Tensor0StrideAxpbyBF16V1ExecuteHandler(),
                ),
                ("execute_axpby_f32", Tensor0StrideAxpbyF32V1ExecuteHandler()),
                ("execute_axpby_f64", Tensor0StrideAxpbyF64V1ExecuteHandler()),
                ("execute_axpby_c64", Tensor0StrideAxpbyC64V1ExecuteHandler()),
                (
                    "execute_axpby_c128",
                    Tensor0StrideAxpbyC128V1ExecuteHandler(),
                ),
                ("execute_dotu_f32", Tensor0StrideDotuF32V1ExecuteHandler()),
                ("execute_dotc_f32", Tensor0StrideDotcF32V1ExecuteHandler()),
                ("execute_dotu_c64", Tensor0StrideDotuC64V1ExecuteHandler()),
                ("execute_dotc_c64", Tensor0StrideDotcC64V1ExecuteHandler()),
                (
                    "execute_selected_scale_alias_f32",
                    Tensor0StrideSelectedScaleAliasF32V1ExecuteHandler(),
                ),
                (
                    "execute_selected_scale_alias_c64",
                    Tensor0StrideSelectedScaleAliasC64V1ExecuteHandler(),
                ),
                (
                    "instantiate_structured_reduction",
                    Tensor0StrideStructuredReductionV1InstantiateHandler(),
                ),
                (
                    "instantiate_grouped_reduction",
                    Tensor0StrideGroupedReductionV1InstantiateHandler(),
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
pub fn _observe_stride_leaf_kernels_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideObserveLeafKernelsForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _stride_leaf_kernel_masks_for_tests() -> (u64, u64) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        (Tensor0StrideLeafKernelMaskForTests(), Tensor0StrideSupportedLeafKernelMaskForTests())
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        (0, 0)
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
pub fn _stride_last_reduction_fiber_chunks() -> u64 {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StrideLastReductionFiberChunks() }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        0
    }
}

#[pyfunction]
pub fn _stride_last_grouped_output_owner() -> bool {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe { Tensor0StrideLastGroupedOutputOwner() != 0 }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        false
    }
}

#[pyfunction]
pub fn _stride_alias_pointers() -> (u64, u64, u64, u64) {
    #[cfg(tensor0_stride_ffi)]
    {
        unsafe {
            (
                Tensor0StrideAliasLastBasePointer(),
                Tensor0StrideAliasLastSourcePointer(),
                Tensor0StrideAliasLastFactorPointer(),
                Tensor0StrideAliasLastResultPointer(),
            )
        }
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        (0, 0, 0, 0)
    }
}

#[pyfunction]
pub fn _reset_stride_alias_pointers() {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideResetAliasPointers();
    }
}

#[pyfunction]
pub fn _stride_worker_limit() -> Option<u64> {
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
pub fn _set_stride_worker_limit(limit: Option<u64>) {
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
pub fn _set_stride_force_generic_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetForceGenericForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _set_stride_force_generated_baseline_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetForceGeneratedBaselineForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _set_stride_reduction_fiber_parallel_mode_for_tests(mode: u64) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetReductionFiberParallelModeForTests(mode);
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = mode;
    }
}

#[pyfunction]
pub fn _set_stride_disable_f16_f32_contiguous_simd_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetDisableF16F32ContiguousSimdForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _set_stride_disable_f16_f16_contiguous_simd_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetDisableF16F16ContiguousSimdForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _set_stride_disable_f32_c64_contiguous_simd_for_tests(enabled: bool) {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideSetDisableF32C64ContiguousSimdForTests(u64::from(enabled));
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        let _ = enabled;
    }
}

#[pyfunction]
pub fn _stride_cpu_supports_f16_contiguous_simd() -> bool {
    #[cfg(tensor0_stride_ffi)]
    unsafe {
        Tensor0StrideCpuSupportsF16ContiguousSimd() != 0
    }
    #[cfg(not(tensor0_stride_ffi))]
    {
        false
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
    module.add_function(wrap_pyfunction!(_observe_stride_leaf_kernels_for_tests, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_leaf_kernel_masks_for_tests, module)?)?;
    module.add_function(wrap_pyfunction!(_reset_stride_native_call_count, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_last_worker_count, module)?)?;
    module.add_function(wrap_pyfunction!(
        _stride_last_available_worker_count,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _stride_last_reduction_fiber_chunks,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_stride_last_grouped_output_owner, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_alias_pointers, module)?)?;
    module.add_function(wrap_pyfunction!(_reset_stride_alias_pointers, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(_set_stride_worker_limit, module)?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_force_generic_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_force_generated_baseline_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_reduction_fiber_parallel_mode_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_disable_f16_f16_contiguous_simd_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_disable_f16_f32_contiguous_simd_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _set_stride_disable_f32_c64_contiguous_simd_for_tests,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        _stride_cpu_supports_f16_contiguous_simd,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_build_versions, module)?)?;
    module.add_function(wrap_pyfunction!(_stride_ffi_abi_version, module)?)?;
    Ok(())
}
