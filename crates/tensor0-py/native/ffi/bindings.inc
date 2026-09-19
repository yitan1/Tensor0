XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideDotInstantiateV1, tensor0::stride::InstantiateDot,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<int64_t>("conjugate_left"));

extern "C" void* Tensor0StrideDotInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideDotInstantiateV1);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideInstantiateV1, tensor0::stride::InstantiatePrepared,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout"));

extern "C" void* Tensor0StrideInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideInstantiateV1);
}

extern "C" void* Tensor0StridePreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::PreparedState::id);
}

extern "C" const void* Tensor0StridePreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::kPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StridePreparedCreatedCount() {
  return tensor0::stride::prepared_created_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedDestroyedCount() {
  return tensor0::stride::prepared_destroyed_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideGetWorkerLimit() {
  return tensor0::stride::worker_limit.load(std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetWorkerLimit(uint64_t limit) {
  tensor0::stride::worker_limit.store(limit, std::memory_order_relaxed);
}

extern "C" const char* Tensor0StrideBuiltJaxVersion() {
  return TENSOR0_STRIDE_JAX_VERSION;
}

extern "C" const char* Tensor0StrideBuiltJaxlibVersion() {
  return TENSOR0_STRIDE_JAXLIB_VERSION;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideReductionInstantiateV1, tensor0::stride::InstantiateReduction,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const uint8_t>>("layout")
        .Attr<ffi::Span<const int64_t>>("coefficient_records"));

extern "C" void* Tensor0StrideReductionInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideReductionInstantiateV1);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAccumulationInstantiateV1, tensor0::stride::InstantiateAccumulation,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<ffi::Span<const int64_t>>("coefficient_records"));

extern "C" void* Tensor0StrideAccumulationInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideAccumulationInstantiateV1);
}

#define TENSOR0_STRIDE_DEFINE_COPY(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCopy##Suffix##V1, tensor0::stride::Copy<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideCopy##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCopy##Suffix##V1); \
}

#define TENSOR0_STRIDE_DEFINE_UPDATE(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideUpdate##Suffix##V1, tensor0::stride::Update<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR2<ffi::Dtype>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideUpdate##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideUpdate##Suffix##V1); \
}

#define TENSOR0_STRIDE_DEFINE_REDUCTION(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideReduction##Suffix##V1, tensor0::stride::Reduction<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const uint8_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().RemainingArgs().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideReduction##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideReduction##Suffix##V1); \
}

#define TENSOR0_STRIDE_DEFINE_ACCUMULATION(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideAccumulation##Suffix##V1, tensor0::stride::Accumulation<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().RemainingArgs().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideAccumulation##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideAccumulation##Suffix##V1); \
}

#define TENSOR0_STRIDE_DEFINE_DOT(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideDot##Suffix##V1, tensor0::stride::Dot<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<int64_t>("conjugate_left") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>() \
        .Ret<ffi::BufferR1<ffi::Dtype>>().Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideDot##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideDot##Suffix##V1); \
}

#define TENSOR0_STRIDE_FOR_EACH_DTYPE(Define) \
  Define(S32, S32) \
  Define(F32, F32) \
  Define(F16, F16) \
  Define(BF16, BF16) \
  Define(C64, C64) \
  Define(F64, F64) \
  Define(C128, C128) \
  Define(S64, S64) \
  Define(U64, U64) \
  Define(S16, S16) \
  Define(S8, S8) \
  Define(U8, U8) \
  Define(U16, U16) \
  Define(U32, U32) \
  Define(Pred, PRED)

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_COPY)
TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_UPDATE)
TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_REDUCTION)
TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_ACCUMULATION)

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_DOT)

#undef TENSOR0_STRIDE_FOR_EACH_DTYPE
#undef TENSOR0_STRIDE_DEFINE_COPY
#undef TENSOR0_STRIDE_DEFINE_UPDATE
#undef TENSOR0_STRIDE_DEFINE_REDUCTION
#undef TENSOR0_STRIDE_DEFINE_ACCUMULATION
#undef TENSOR0_STRIDE_DEFINE_DOT
