#include "prepared.h"

#include "errors.h"
#include <stdexcept>
#include <utility>

namespace tensor0::stride {

std::atomic<uint64_t> prepared_created_count{0};
std::atomic<uint64_t> prepared_destroyed_count{0};

PreparedState::PreparedState(descriptor::DecodedLayout value)
    : records(std::move(value.records)), source_size(value.source_size), output_size(value.output_size) {
  prepared_created_count.fetch_add(1, std::memory_order_relaxed);
}

PreparedState::~PreparedState() {
  prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
}

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(ffi::Span<const int64_t> words) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto layout = descriptor::PrepareLayout(words.begin(), words.size());
    state = std::make_unique<PreparedState>(std::move(layout));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateReduction(
    ffi::Span<const uint8_t> bytes, ffi::Span<const int64_t>) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto decoded = descriptor::PrepareReductionLayout(
        reinterpret_cast<const char*>(bytes.begin()), bytes.size());
    descriptor::DecodedLayout legacy;
    legacy.source_size = decoded.source_size;
    legacy.output_size = decoded.output_size;
    legacy.records.reserve(decoded.records.size());
    // Fuse role-aware records before projecting their address layouts.
    for (auto& record : decoded.records) legacy.records.push_back(std::move(record.layout));
    state = std::make_unique<PreparedState>(std::move(legacy));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateDot(
    ffi::Span<const int64_t> words, int64_t) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto layout = descriptor::PrepareAddressLayout(words.begin(), words.size());
    state = std::make_unique<PreparedState>(std::move(layout));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateAccumulation(
    ffi::Span<const int64_t> words, ffi::Span<const int64_t>) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto layout = descriptor::PrepareAccumulationLayout(words.begin(), words.size());
    state = std::make_unique<PreparedState>(std::move(layout));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

std::shared_ptr<const std::vector<layout::GeneratedRecordProgram>> PreparedState::AccumulationPrograms(
    uint64_t source_item_size, uint64_t result_item_size) const {
  std::lock_guard lock(accumulation_mutex);
  const auto key = std::make_pair(source_item_size, result_item_size);
  if (auto found = accumulation_programs.find(key); found != accumulation_programs.end()) {
    return found->second;
  }
  auto programs = std::make_shared<const std::vector<layout::GeneratedRecordProgram>>(
      layout::PrepareGeneratedRecords(records, source_item_size, result_item_size, false));
  accumulation_programs.emplace(key, programs);
  return programs;
}

void ValidatePreparedDimensions(const PreparedState& prepared, uint64_t source_size, uint64_t output_size) {
  if (prepared.source_size != source_size || prepared.output_size != output_size) {
    throw std::invalid_argument("buffer dimensions do not match layout");
  }
}

}

#ifndef TENSOR0_STRIDE_JAX_VERSION
#define TENSOR0_STRIDE_JAX_VERSION "unknown"
#endif

#ifndef TENSOR0_STRIDE_JAXLIB_VERSION
#define TENSOR0_STRIDE_JAXLIB_VERSION "unknown"
#endif

namespace ffi = xla::ffi;

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
