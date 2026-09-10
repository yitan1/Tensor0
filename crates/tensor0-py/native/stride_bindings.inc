
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedF32V6Execute,
    tensor0::stride::ExecutePreparedF32Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedF16V6Execute,
    tensor0::stride::ExecutePreparedF16Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedBF16V6Execute,
    tensor0::stride::ExecutePreparedBF16Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedC64V6Execute,
    tensor0::stride::ExecutePreparedC64Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::C64>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedS32V6Execute,
    tensor0::stride::ExecutePreparedS32Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ctx<ffi::ThreadPool>());

#define TENSOR0_DEFINE_PREPARED_EXECUTE(Symbol, Function, Dtype)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Ret<ffi::Buffer<Dtype>>()                                         \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedPredV6Execute,
                                tensor0::stride::ExecutePreparedPredImpl,
                                ffi::PRED)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedS8V6Execute,
                                tensor0::stride::ExecutePreparedS8Impl,
                                ffi::S8)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedS16V6Execute,
                                tensor0::stride::ExecutePreparedS16Impl,
                                ffi::S16)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedS64V6Execute,
                                tensor0::stride::ExecutePreparedS64Impl,
                                ffi::S64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedU8V6Execute,
                                tensor0::stride::ExecutePreparedU8Impl,
                                ffi::U8)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedU16V6Execute,
                                tensor0::stride::ExecutePreparedU16Impl,
                                ffi::U16)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedU32V6Execute,
                                tensor0::stride::ExecutePreparedU32Impl,
                                ffi::U32)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedU64V6Execute,
                                tensor0::stride::ExecutePreparedU64Impl,
                                ffi::U64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedF64V6Execute,
                                tensor0::stride::ExecutePreparedF64Impl,
                                ffi::F64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideAffinePreparedC128V6Execute,
                                tensor0::stride::ExecutePreparedC128Impl,
                                ffi::C128)

#undef TENSOR0_DEFINE_PREPARED_EXECUTE

#define TENSOR0_DEFINE_MIXED_EXECUTE(Symbol, Function, SourceType, ResultType) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                           \
      Symbol, Function,                                                    \
      ffi::Ffi::BindExecute()                                              \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                    \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()               \
          .Arg<ffi::Buffer<SourceType>>()                                  \
          .Ret<ffi::Buffer<ResultType>>()                                  \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF16F32ForwardV1Execute,
    tensor0::stride::ExecutePreparedF16F32ForwardImpl, ffi::F16, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32F16TransposeV1Execute,
    tensor0::stride::ExecutePreparedF32F16TransposeImpl, ffi::F32, ffi::F16)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32C64ForwardV1Execute,
    tensor0::stride::ExecutePreparedF32C64ForwardImpl, ffi::F32, ffi::C64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC64F32TransposeV1Execute,
    tensor0::stride::ExecutePreparedC64F32TransposeImpl, ffi::C64, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC64F32ForwardV1Execute,
    tensor0::stride::ExecutePreparedC64F32ForwardImpl, ffi::C64, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32C64TransposeV1Execute,
    tensor0::stride::ExecutePreparedF32C64TransposeImpl, ffi::F32, ffi::C64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF64C128ForwardV1Execute,
    tensor0::stride::ExecutePreparedF64C128ForwardImpl, ffi::F64, ffi::C128)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC128F64TransposeV1Execute,
    tensor0::stride::ExecutePreparedC128F64TransposeImpl, ffi::C128, ffi::F64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC128F64ForwardV1Execute,
    tensor0::stride::ExecutePreparedC128F64ForwardImpl, ffi::C128, ffi::F64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF64C128TransposeV1Execute,
    tensor0::stride::ExecutePreparedF64C128TransposeImpl, ffi::F64, ffi::C128)

#undef TENSOR0_DEFINE_MIXED_EXECUTE

#define TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(Symbol, Function, Dtype)       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Arg<ffi::AnyBuffer>()                                         \
          .Ret<ffi::Buffer<Dtype>>()                                         \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleAliasF32V1Execute,
    tensor0::stride::ExecuteSelectedScaleAliasF32Impl, ffi::F32)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleAliasC64V1Execute,
    tensor0::stride::ExecuteSelectedScaleAliasC64Impl, ffi::C64)

TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentF16V1Execute,
    tensor0::stride::ExecuteScaleTangentF16Impl, ffi::F16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentBF16V1Execute,
    tensor0::stride::ExecuteScaleTangentBF16Impl, ffi::BF16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentF32V1Execute,
    tensor0::stride::ExecuteScaleTangentF32Impl, ffi::F32)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentF64V1Execute,
    tensor0::stride::ExecuteScaleTangentF64Impl, ffi::F64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentC64V1Execute,
    tensor0::stride::ExecuteScaleTangentC64Impl, ffi::C64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideScaleTangentC128V1Execute,
    tensor0::stride::ExecuteScaleTangentC128Impl, ffi::C128)

#undef TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE

#define TENSOR0_DEFINE_REDUCTION_EXECUTE(Symbol, Function, SourceType,       \
                                         ResultType)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<SourceType>>()                                    \
          .Ret<ffi::Buffer<ResultType>>()                                    \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32Impl, ffi::F32, ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16V1Execute,
    tensor0::stride::ExecuteStructuredReductionF16Impl, ffi::F16, ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionBF16V1Execute,
    tensor0::stride::ExecuteStructuredReductionBF16Impl, ffi::BF16,
    ffi::BF16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64V1Execute,
    tensor0::stride::ExecuteStructuredReductionC64Impl, ffi::C64, ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32F16V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32F16Impl, ffi::F32,
    ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64F32V1Execute,
    tensor0::stride::ExecuteStructuredReductionC64F32Impl, ffi::C64,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32C64V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32C64Impl, ffi::F32,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64V1Execute,
    tensor0::stride::ExecuteStructuredReductionF64Impl, ffi::F64, ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128V1Execute,
    tensor0::stride::ExecuteStructuredReductionC128Impl, ffi::C128,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128F64V1Execute,
    tensor0::stride::ExecuteStructuredReductionC128F64Impl, ffi::C128,
    ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64C128V1Execute,
    tensor0::stride::ExecuteStructuredReductionF64C128Impl, ffi::F64,
    ffi::C128)

TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF32ForwardImpl, ffi::F32,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF16ForwardImpl, ffi::F16,
    ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionBF16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionBF16ForwardImpl, ffi::BF16,
    ffi::BF16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC64ForwardImpl, ffi::C64,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionPredForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionPredForwardImpl, ffi::PRED,
    ffi::PRED)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS8ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS8ForwardImpl, ffi::S8,
    ffi::S8)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS16ForwardImpl, ffi::S16,
    ffi::S16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS32ForwardImpl, ffi::S32,
    ffi::S32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS64ForwardImpl, ffi::S64,
    ffi::S64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU8ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU8ForwardImpl, ffi::U8,
    ffi::U8)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU16ForwardImpl, ffi::U16,
    ffi::U16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU32ForwardImpl, ffi::U32,
    ffi::U32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU64ForwardImpl, ffi::U64,
    ffi::U64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16F32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF16F32ForwardImpl, ffi::F16,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32C64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF32C64ForwardImpl, ffi::F32,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64F32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC64F32ForwardImpl, ffi::C64,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF64ForwardImpl, ffi::F64,
    ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC128ForwardImpl, ffi::C128,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64C128ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF64C128ForwardImpl, ffi::F64,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128F64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC128F64ForwardImpl, ffi::C128,
    ffi::F64)

#undef TENSOR0_DEFINE_REDUCTION_EXECUTE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAffinePreparedF32V6Instantiate,
    tensor0::stride::InstantiatePreparedImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideSelectedScaleV1Instantiate,
    tensor0::stride::InstantiatePreparedOperationImpl<
        tensor0::stride::PreparedOperation::kSelectedScale>,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

#define TENSOR0_DEFINE_AXPBY_EXECUTE(Symbol, Function, Dtype)                \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Attr<ffi::Span<const uint64_t>>("stages").Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Arg<ffi::AnyBuffer>()                                         \
          .Arg<ffi::AnyBuffer>()                                         \
          .Ret<ffi::Buffer<Dtype>>()                                         \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyF16V1Execute,
                             tensor0::stride::ExecuteAxpbyF16Impl, ffi::F16)
TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyBF16V1Execute,
                             tensor0::stride::ExecuteAxpbyBF16Impl, ffi::BF16)
TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyF32V1Execute,
                             tensor0::stride::ExecuteAxpbyF32Impl, ffi::F32)
TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyF64V1Execute,
                             tensor0::stride::ExecuteAxpbyF64Impl, ffi::F64)
TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyC64V1Execute,
                             tensor0::stride::ExecuteAxpbyC64Impl, ffi::C64)
TENSOR0_DEFINE_AXPBY_EXECUTE(Tensor0StrideAxpbyC128V1Execute,
                             tensor0::stride::ExecuteAxpbyC128Impl, ffi::C128)

#undef TENSOR0_DEFINE_AXPBY_EXECUTE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideDotV1Instantiate,
    tensor0::stride::InstantiatePreparedDotImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

#define TENSOR0_DEFINE_DOT_EXECUTE(Symbol, Function, Dtype)                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                            \
      Symbol, Function,                                                     \
      ffi::Ffi::BindExecute()                                               \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                    \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                \
          .Arg<ffi::Buffer<Dtype>>()                                        \
          .Arg<ffi::Buffer<Dtype>>()                                        \
          .Ret<ffi::Buffer<Dtype>>()                                        \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_DOT_EXECUTE(Tensor0StrideDotuF32V1Execute,
                            tensor0::stride::ExecuteDotuF32Impl, ffi::F32)
TENSOR0_DEFINE_DOT_EXECUTE(Tensor0StrideDotcF32V1Execute,
                            tensor0::stride::ExecuteDotcF32Impl, ffi::F32)
TENSOR0_DEFINE_DOT_EXECUTE(Tensor0StrideDotuC64V1Execute,
                            tensor0::stride::ExecuteDotuC64Impl, ffi::C64)
TENSOR0_DEFINE_DOT_EXECUTE(Tensor0StrideDotcC64V1Execute,
                            tensor0::stride::ExecuteDotcC64Impl, ffi::C64)

#undef TENSOR0_DEFINE_DOT_EXECUTE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideStructuredReductionV1Instantiate,
    tensor0::stride::InstantiatePreparedReductionImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideGroupedReductionV1Instantiate,
    tensor0::stride::InstantiatePreparedGroupedReductionImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

extern "C" void* Tensor0StrideAffinePreparedF32V6InstantiateHandler() {
  return reinterpret_cast<void*>(
      &Tensor0StrideAffinePreparedF32V6Instantiate);
}

extern "C" void* Tensor0StrideAffinePreparedF32V6ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAffinePreparedF32V6Execute);
}

extern "C" void* Tensor0StrideAffinePreparedF16V6ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAffinePreparedF16V6Execute);
}

extern "C" void* Tensor0StrideAffinePreparedBF16V6ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAffinePreparedBF16V6Execute);
}

extern "C" void* Tensor0StrideAffinePreparedC64V6ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAffinePreparedC64V6Execute);
}

extern "C" void* Tensor0StrideAffinePreparedS32V6ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAffinePreparedS32V6Execute);
}

#define TENSOR0_PREPARED_HANDLER(Name, Symbol) \
  extern "C" void* Name() { return reinterpret_cast<void*>(&Symbol); }

TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedPredV6ExecuteHandler,
                         Tensor0StrideAffinePreparedPredV6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedS8V6ExecuteHandler,
                         Tensor0StrideAffinePreparedS8V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedS16V6ExecuteHandler,
                         Tensor0StrideAffinePreparedS16V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedS64V6ExecuteHandler,
                         Tensor0StrideAffinePreparedS64V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedU8V6ExecuteHandler,
                         Tensor0StrideAffinePreparedU8V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedU16V6ExecuteHandler,
                         Tensor0StrideAffinePreparedU16V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedU32V6ExecuteHandler,
                         Tensor0StrideAffinePreparedU32V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedU64V6ExecuteHandler,
                         Tensor0StrideAffinePreparedU64V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedF64V6ExecuteHandler,
                         Tensor0StrideAffinePreparedF64V6Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideAffinePreparedC128V6ExecuteHandler,
                         Tensor0StrideAffinePreparedC128V6Execute)

#undef TENSOR0_PREPARED_HANDLER

#define TENSOR0_MIXED_HANDLER(Name, Symbol)       \
  extern "C" void* Name() {                      \
    return reinterpret_cast<void*>(&Symbol);     \
  }

TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF16F32ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF16F32ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32F16TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF32F16TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32C64ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF32C64ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC64F32TransposeV1ExecuteHandler,
                      Tensor0StrideAffineC64F32TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC64F32ForwardV1ExecuteHandler,
                      Tensor0StrideAffineC64F32ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32C64TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF32C64TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF64C128ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF64C128ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC128F64TransposeV1ExecuteHandler,
                      Tensor0StrideAffineC128F64TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC128F64ForwardV1ExecuteHandler,
                      Tensor0StrideAffineC128F64ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF64C128TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF64C128TransposeV1Execute)

#undef TENSOR0_MIXED_HANDLER

extern "C" void* Tensor0StrideSelectedScaleV1InstantiateHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideSelectedScaleV1Instantiate);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideUpdateV1Instantiate,
    tensor0::stride::InstantiateTypedUpdate,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Attr<ffi::Span<const uint64_t>>("stages"));

extern "C" void* Tensor0StrideUpdateV1InstantiateHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateV1Instantiate);
}

extern "C" void* Tensor0StrideAxpbyF32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyF32V1Execute);
}

extern "C" void* Tensor0StrideAxpbyF16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyF16V1Execute);
}

extern "C" void* Tensor0StrideAxpbyBF16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyBF16V1Execute);
}

extern "C" void* Tensor0StrideAxpbyF64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyF64V1Execute);
}

extern "C" void* Tensor0StrideAxpbyC64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyC64V1Execute);
}

extern "C" void* Tensor0StrideAxpbyC128V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideAxpbyC128V1Execute);
}

#define TENSOR0_EXECUTE_HANDLER(Name, Symbol) \
  extern "C" void* Name() {                     \
    return reinterpret_cast<void*>(&Symbol);     \
  }

TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentF16V1ExecuteHandler,
                            Tensor0StrideScaleTangentF16V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentBF16V1ExecuteHandler,
                            Tensor0StrideScaleTangentBF16V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentF32V1ExecuteHandler,
                            Tensor0StrideScaleTangentF32V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentF64V1ExecuteHandler,
                            Tensor0StrideScaleTangentF64V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentC64V1ExecuteHandler,
                            Tensor0StrideScaleTangentC64V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideScaleTangentC128V1ExecuteHandler,
                            Tensor0StrideScaleTangentC128V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideSelectedScaleAliasF32V1ExecuteHandler,
    Tensor0StrideSelectedScaleAliasF32V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideSelectedScaleAliasC64V1ExecuteHandler,
    Tensor0StrideSelectedScaleAliasC64V1Execute)

TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionV1InstantiateHandler,
    Tensor0StrideStructuredReductionV1Instantiate)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideGroupedReductionV1InstantiateHandler,
    Tensor0StrideGroupedReductionV1Instantiate)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideDotV1InstantiateHandler,
                            Tensor0StrideDotV1Instantiate)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideDotuF32V1ExecuteHandler,
                            Tensor0StrideDotuF32V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideDotcF32V1ExecuteHandler,
                            Tensor0StrideDotcF32V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideDotuC64V1ExecuteHandler,
                            Tensor0StrideDotuC64V1Execute)
TENSOR0_EXECUTE_HANDLER(Tensor0StrideDotcC64V1ExecuteHandler,
                            Tensor0StrideDotcC64V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF32V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF16V1ExecuteHandler,
    Tensor0StrideStructuredReductionF16V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionBF16V1ExecuteHandler,
    Tensor0StrideStructuredReductionBF16V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC64V1ExecuteHandler,
    Tensor0StrideStructuredReductionC64V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF32F16V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32F16V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC64F32V1ExecuteHandler,
    Tensor0StrideStructuredReductionC64F32V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF32C64V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32C64V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF64V1ExecuteHandler,
    Tensor0StrideStructuredReductionF64V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC128V1ExecuteHandler,
    Tensor0StrideStructuredReductionC128V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC128F64V1ExecuteHandler,
    Tensor0StrideStructuredReductionC128F64V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF64C128V1ExecuteHandler,
    Tensor0StrideStructuredReductionF64C128V1Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF32ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF16ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionBF16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionBF16ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC64ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionPredForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionPredForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionS8ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS8ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionS16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS16ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionS32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS32ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionS64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS64ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionU8ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU8ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionU16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU16ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionU32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU32ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionU64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU64ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF16F32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF16F32ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF32C64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF32C64ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC64F32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC64F32ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF64ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC128ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC128ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionF64C128ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF64C128ForwardV2Execute)
TENSOR0_EXECUTE_HANDLER(
    Tensor0StrideStructuredReductionC128F64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC128F64ForwardV2Execute)

#undef TENSOR0_EXECUTE_HANDLER

extern "C" void* Tensor0StridePreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::PreparedState::id);
}

extern "C" const void* Tensor0StridePreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::kPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StridePreparedInstantiateCount() {
  return tensor0::stride::prepared_instantiate_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedExecuteCount() {
  return tensor0::stride::prepared_execute_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLiveStateCount() {
  return tensor0::stride::prepared_live_state_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedDestroyedStateCount() {
  return tensor0::stride::prepared_destroyed_state_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLiveBytes() {
  return tensor0::stride::prepared_live_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLastStateBytes() {
  return tensor0::stride::prepared_last_state_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLastDescriptorBytes() {
  return tensor0::stride::prepared_last_descriptor_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedResetMetrics() {
  if (tensor0::stride::prepared_live_state_count.load(
          std::memory_order_relaxed) != 0) {
    return 0;
  }
  tensor0::stride::prepared_instantiate_count.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_execute_count.store(0,
                                                std::memory_order_relaxed);
  tensor0::stride::prepared_destroyed_state_count.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_live_bytes.store(0,
                                             std::memory_order_relaxed);
  tensor0::stride::prepared_last_state_bytes.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_last_descriptor_bytes.store(
      0, std::memory_order_relaxed);
  return 1;
}

extern "C" uint64_t Tensor0StrideNativeCallCount() {
  return tensor0::stride::native_call_count.load(std::memory_order_relaxed);
}

extern "C" void Tensor0StrideObserveLeafKernelsForTests(uint64_t enabled) {
  tensor0::stride::leaf_kernel_mask_for_tests.store(0, std::memory_order_relaxed);
  tensor0::stride::observe_leaf_kernels_for_tests.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLeafKernelMaskForTests() {
  return tensor0::stride::leaf_kernel_mask_for_tests.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideSupportedLeafKernelMaskForTests() {
  uint64_t mask = 12;
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (tensor0::stride::CpuSupportsAvx2F16c()) mask |= 3;
#endif
  return mask;
}

extern "C" void Tensor0StrideResetNativeCallCount() {
  tensor0::stride::native_call_count.store(0, std::memory_order_relaxed);
  tensor0::stride::last_worker_count.store(0, std::memory_order_relaxed);
  tensor0::stride::last_available_worker_count.store(
      0, std::memory_order_relaxed);
  tensor0::stride::last_reduction_fiber_chunks.store(
      0, std::memory_order_relaxed);
  tensor0::stride::last_grouped_output_owner.store(
      false, std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastWorkerCount() {
  return tensor0::stride::last_worker_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastAvailableWorkerCount() {
  return tensor0::stride::last_available_worker_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastReductionFiberChunks() {
  return tensor0::stride::last_reduction_fiber_chunks.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastGroupedOutputOwner() {
  return tensor0::stride::last_grouped_output_owner.load(
             std::memory_order_relaxed)
             ? 1
             : 0;
}

extern "C" uint64_t Tensor0StrideAliasLastBasePointer() {
  return tensor0::stride::alias_last_base_pointer.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideAliasLastSourcePointer() {
  return tensor0::stride::alias_last_source_pointer.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideAliasLastFactorPointer() {
  return tensor0::stride::alias_last_factor_pointer.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideAliasLastResultPointer() {
  return tensor0::stride::alias_last_result_pointer.load(
      std::memory_order_relaxed);
}

extern "C" void Tensor0StrideResetAliasPointers() {
  tensor0::stride::alias_last_base_pointer.store(0,
                                                  std::memory_order_relaxed);
  tensor0::stride::alias_last_source_pointer.store(
      0, std::memory_order_relaxed);
  tensor0::stride::alias_last_factor_pointer.store(
      0, std::memory_order_relaxed);
  tensor0::stride::alias_last_result_pointer.store(
      0, std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideGetWorkerLimit() {
  return tensor0::stride::worker_limit.load(std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetWorkerLimit(uint64_t limit) {
  tensor0::stride::worker_limit.store(limit, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetForceGenericForTests(uint64_t enabled) {
  tensor0::stride::force_generic_for_tests.store(
      enabled != 0, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetForceGeneratedBaselineForTests(
    uint64_t enabled) {
  tensor0::stride::force_generated_baseline_for_tests.store(
      enabled != 0, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetReductionFiberParallelModeForTests(
    uint64_t mode) {
  tensor0::stride::reduction_fiber_parallel_mode_for_tests.store(
      mode, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetDisableF16F32ContiguousSimdForTests(
    uint64_t enabled) {
  tensor0::stride::disable_f16_f32_contiguous_simd_for_tests.store(
      enabled != 0, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetDisableF16F16ContiguousSimdForTests(
    uint64_t enabled) {
  tensor0::stride::disable_f16_f16_contiguous_simd_for_tests.store(
      enabled != 0, std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetDisableF32C64ContiguousSimdForTests(
    uint64_t enabled) {
  tensor0::stride::disable_f32_c64_contiguous_simd_for_tests.store(
      enabled != 0, std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideCpuSupportsF16ContiguousSimd() {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  return tensor0::stride::CpuSupportsAvx2F16c() ? 1 : 0;
#else
  return 0;
#endif
}

extern "C" const char* Tensor0StrideBuiltJaxVersion() {
  return TENSOR0_STRIDE_JAX_VERSION;
}

extern "C" const char* Tensor0StrideBuiltJaxlibVersion() {
  return TENSOR0_STRIDE_JAXLIB_VERSION;
}

extern "C" uint64_t Tensor0StrideAbiVersion() {
  return tensor0::stride::kDescriptorVersion;
}

// XLA FFI handler definitions and exported registration/diagnostic symbols.


#define TENSOR0_DEFINE_UPDATE_EXECUTE(Symbol, Function, Source, Dtype)      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                            \
      Symbol, Function,                                                    \
      ffi::Ffi::BindExecute().Attr<ffi::Span<const uint8_t>>("descriptor")           \
          .Attr<ffi::Span<const uint64_t>>("stages").Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>().Arg<ffi::Buffer<Source>>()              \
          .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()               \
          .Ret<ffi::Buffer<Dtype>>().Ctx<ffi::ThreadPool>());                 
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF32V1Execute,
    tensor0::stride::ExecuteUpdateF32Impl, ffi::F32, ffi::F32)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF16V1Execute,
    tensor0::stride::ExecuteUpdateF16Impl, ffi::F16, ffi::F16)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateBF16V1Execute,
    tensor0::stride::ExecuteUpdateBF16Impl, ffi::BF16, ffi::BF16)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateC64V1Execute,
    tensor0::stride::ExecuteUpdateC64Impl, ffi::C64, ffi::C64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS32V1Execute,
    tensor0::stride::ExecuteUpdateS32Impl, ffi::S32, ffi::S32)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdatePredV1Execute,
    tensor0::stride::ExecuteUpdatePredImpl, ffi::PRED, ffi::PRED)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS8V1Execute,
    tensor0::stride::ExecuteUpdateS8Impl, ffi::S8, ffi::S8)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS16V1Execute,
    tensor0::stride::ExecuteUpdateS16Impl, ffi::S16, ffi::S16)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS64V1Execute,
    tensor0::stride::ExecuteUpdateS64Impl, ffi::S64, ffi::S64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateU8V1Execute,
    tensor0::stride::ExecuteUpdateU8Impl, ffi::U8, ffi::U8)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateU16V1Execute,
    tensor0::stride::ExecuteUpdateU16Impl, ffi::U16, ffi::U16)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateU32V1Execute,
    tensor0::stride::ExecuteUpdateU32Impl, ffi::U32, ffi::U32)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateU64V1Execute,
    tensor0::stride::ExecuteUpdateU64Impl, ffi::U64, ffi::U64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF64V1Execute,
    tensor0::stride::ExecuteUpdateF64Impl, ffi::F64, ffi::F64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateC128V1Execute,
    tensor0::stride::ExecuteUpdateC128Impl, ffi::C128, ffi::C128)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF16F32V1Execute,
    tensor0::stride::ExecuteUpdateF16F32Impl, ffi::F16, ffi::F32)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF32C64V1Execute,
    tensor0::stride::ExecuteUpdateF32C64Impl, ffi::F32, ffi::C64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateC64F32V1Execute,
    tensor0::stride::ExecuteUpdateC64F32Impl, ffi::C64, ffi::F32)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateF64C128V1Execute,
    tensor0::stride::ExecuteUpdateF64C128Impl, ffi::F64, ffi::C128)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateC128F64V1Execute,
    tensor0::stride::ExecuteUpdateC128F64Impl, ffi::C128, ffi::F64)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS32PredV1Execute,
    tensor0::stride::ExecuteUpdateS32PredImpl, ffi::S32, ffi::PRED)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS32S8V1Execute,
    tensor0::stride::ExecuteUpdateS32S8Impl, ffi::S32, ffi::S8)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateS32S16V1Execute,
    tensor0::stride::ExecuteUpdateS32S16Impl, ffi::S32, ffi::S16)
TENSOR0_DEFINE_UPDATE_EXECUTE(Tensor0StrideUpdateU32U8V1Execute,
    tensor0::stride::ExecuteUpdateU32U8Impl, ffi::U32, ffi::U8)

#undef TENSOR0_DEFINE_UPDATE_EXECUTE
extern "C" void* Tensor0StrideUpdateF32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF32V1Execute);
}
extern "C" void* Tensor0StrideUpdateF16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF16V1Execute);
}
extern "C" void* Tensor0StrideUpdateBF16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateBF16V1Execute);
}
extern "C" void* Tensor0StrideUpdateC64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateC64V1Execute);
}
extern "C" void* Tensor0StrideUpdateS32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS32V1Execute);
}
extern "C" void* Tensor0StrideUpdatePredV1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdatePredV1Execute);
}
extern "C" void* Tensor0StrideUpdateS8V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS8V1Execute);
}
extern "C" void* Tensor0StrideUpdateS16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS16V1Execute);
}
extern "C" void* Tensor0StrideUpdateS64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS64V1Execute);
}
extern "C" void* Tensor0StrideUpdateU8V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateU8V1Execute);
}
extern "C" void* Tensor0StrideUpdateU16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateU16V1Execute);
}
extern "C" void* Tensor0StrideUpdateU32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateU32V1Execute);
}
extern "C" void* Tensor0StrideUpdateU64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateU64V1Execute);
}
extern "C" void* Tensor0StrideUpdateF64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF64V1Execute);
}
extern "C" void* Tensor0StrideUpdateC128V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateC128V1Execute);
}
extern "C" void* Tensor0StrideUpdateF16F32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF16F32V1Execute);
}
extern "C" void* Tensor0StrideUpdateF32C64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF32C64V1Execute);
}
extern "C" void* Tensor0StrideUpdateC64F32V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateC64F32V1Execute);
}
extern "C" void* Tensor0StrideUpdateF64C128V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateF64C128V1Execute);
}
extern "C" void* Tensor0StrideUpdateC128F64V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateC128F64V1Execute);
}
extern "C" void* Tensor0StrideUpdateS32PredV1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS32PredV1Execute);
}
extern "C" void* Tensor0StrideUpdateS32S8V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS32S8V1Execute);
}
extern "C" void* Tensor0StrideUpdateS32S16V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateS32S16V1Execute);
}
extern "C" void* Tensor0StrideUpdateU32U8V1ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideUpdateU32U8V1Execute);
}
