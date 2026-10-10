#include "execute/dot.cuh"
#include "arithmetic.cuh"
#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"
#include "../layout/owner_fiber.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ffi = xla::ffi;
namespace tensor0::stride::cuda::dot {

namespace {
std::atomic<uint64_t> dot_prepared_created_count{0};
std::atomic<uint64_t> dot_prepared_destroyed_count{0};
}  // namespace

struct DotPreparedState {
  static ffi::TypeId id;

  DotPreparedState(descriptor::DecodedLayout value, std::vector<uint64_t> sizes,
                   std::vector<layout::OwnerFiberSchedule> tasks,
                   uint64_t scratch_capacity, std::size_t words, bool conjugate_left)
      : decoded(std::move(value)), counts(std::move(sizes)),
        schedules(std::move(tasks)), capacity(scratch_capacity), word_count(words), conjugate(conjugate_left) {
    dot_prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }
  DotPreparedState(const DotPreparedState&) = delete;
  DotPreparedState& operator=(const DotPreparedState&) = delete;
  ~DotPreparedState() {
    dot_prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const descriptor::DecodedLayout decoded;
  const std::vector<uint64_t> counts;
  const std::vector<layout::OwnerFiberSchedule> schedules;
  const uint64_t capacity;
  const std::size_t word_count;
  const bool conjugate;
};

ffi::TypeId DotPreparedState::id = {};
const ffi::TypeInfo kDotPreparedTypeInfo = ffi::MakeTypeInfo<DotPreparedState>();

ffi::ErrorOr<std::unique_ptr<DotPreparedState>> InstantiateDot(
    ffi::Span<const int64_t> words, int64_t conjugate) {
  std::unique_ptr<DotPreparedState> state;
  const auto error = ContainErrors([&] {
    if (conjugate != 0 && conjugate != 1) throw std::invalid_argument("CUDA dot conjugate_left must be zero or one");
    auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
    std::vector<uint64_t> counts;
    counts.reserve(decoded.records.size());
    for (const auto& record : decoded.records) {
      const auto count = layout::ElementCount(record);
      counts.push_back(count);
    }
    auto packed = layout::PackOwnerFiber(decoded, 2);
    const auto capacity = layout::DotScratchCapacity(packed.schedules);
    state = std::make_unique<DotPreparedState>(std::move(decoded), std::move(counts),
        std::move(packed.schedules), capacity, packed.words.size(), conjugate != 0);
  });
  if (error.failure()) return ffi::Unexpected(error);
  return std::move(state);
}

template <ffi::DataType Dtype, typename T>
ffi::Error Dot(const DotPreparedState* prepared,
               ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::BufferR1<ffi::S64> descriptor,
               ffi::ResultBufferR1<Dtype> result, ffi::ResultBufferR2<Dtype> scratch,
               cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (left.element_type() != Dtype || right.element_type() != Dtype) throw std::invalid_argument("CUDA dot supports only same-dtype F32/F64/C64/C128 inputs and result");
    if (left.dimensions().size() != 2 || right.dimensions().size() != 2) throw std::invalid_argument("CUDA dot inputs must be rank-two");
    for (const auto dimensions : {left.dimensions(), right.dimensions(), result->dimensions(), scratch->dimensions()}) {
      for (auto dimension : dimensions) if (dimension < 0) throw std::invalid_argument("CUDA dot dimensions must be nonnegative");
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != prepared->word_count) throw std::invalid_argument("CUDA dot descriptor operand shape does not match layout");
    const auto& decoded = prepared->decoded;
    const uint64_t batches = result->dimensions()[0];
    if (static_cast<uint64_t>(left.dimensions()[0]) != batches || static_cast<uint64_t>(right.dimensions()[0]) != batches ||
        static_cast<uint64_t>(left.dimensions()[1]) != decoded.source_size || static_cast<uint64_t>(right.dimensions()[1]) != decoded.output_size ||
        static_cast<uint64_t>(scratch->dimensions()[0]) != batches) throw std::invalid_argument("CUDA dot buffer dimensions do not match layout");
    for (const auto count : prepared->counts) {
      uint64_t total;
      if (!layout::CheckedMultiply(count, batches, &total)) throw std::invalid_argument("CUDA dot logical batch size overflows");
    }
    if (static_cast<uint64_t>(scratch->dimensions()[1]) != prepared->capacity)
      throw std::invalid_argument("CUDA dot scratch capacity does not match layout");
    uint64_t left_elements, right_elements, scratch_elements;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &left_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &right_elements) ||
        !layout::CheckedMultiply(batches, prepared->capacity, &scratch_elements))
      throw std::invalid_argument("CUDA dot batch storage size overflows");
    const auto output_bytes = BufferBytes(batches, sizeof(T));
    const auto left_bytes = BufferBytes(left_elements, sizeof(T));
    const auto right_bytes = BufferBytes(right_elements, sizeof(T));
    const auto scratch_bytes = BufferBytes(scratch_elements, sizeof(T));
    const auto descriptor_bytes = BufferBytes(prepared->word_count, sizeof(int64_t));
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    auto* temporary = reinterpret_cast<T*>(scratch->untyped_data());
    for (const auto target : {std::pair<void*, uint64_t>{output, output_bytes}, {temporary, scratch_bytes}}) {
      ValidateDisjointBuffers(left.untyped_data(), left_bytes, target.first, target.second);
      ValidateDisjointBuffers(right.untyped_data(), right_bytes, target.first, target.second);
      ValidateDisjointBuffers(descriptor.typed_data(), descriptor_bytes, target.first, target.second);
    }
    ValidateDisjointBuffers(output, output_bytes, temporary, scratch_bytes);
    Execute<T>(reinterpret_cast<const T*>(left.untyped_data()),
        reinterpret_cast<const T*>(right.untyped_data()), output, temporary, prepared->conjugate,
        decoded, prepared->schedules, prepared->capacity, batches, output_bytes,
        descriptor.typed_data(), stream);
  });
}
}  // namespace tensor0::stride::cuda::dot

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideCudaDotInstantiateV1, tensor0::stride::cuda::dot::InstantiateDot,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<int64_t>("conjugate_left"));

extern "C" void* Tensor0StrideCudaDotInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideCudaDotInstantiateV1);
}

extern "C" void* Tensor0StrideCudaDotPreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::cuda::dot::DotPreparedState::id);
}

extern "C" const void* Tensor0StrideCudaDotPreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::cuda::dot::kDotPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StrideCudaDotPreparedCreatedCount() {
  return tensor0::stride::cuda::dot::dot_prepared_created_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideCudaDotPreparedDestroyedCount() {
  return tensor0::stride::cuda::dot::dot_prepared_destroyed_count.load(std::memory_order_relaxed);
}

#define TENSOR0_CUDA_DOT(Suffix, Dtype, T) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaDot##Suffix##V1, (tensor0::stride::cuda::dot::Dot<ffi::Dtype, T>), \
    ffi::Ffi::BindExecute().Ctx<ffi::State<tensor0::stride::cuda::dot::DotPreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>() \
        .Ret<ffi::BufferR1<ffi::Dtype>>().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaDot##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaDot##Suffix##V1); \
}
TENSOR0_CUDA_DOT(F32, F32, float)
TENSOR0_CUDA_DOT(F64, F64, double)
TENSOR0_CUDA_DOT(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>)
TENSOR0_CUDA_DOT(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>)
#undef TENSOR0_CUDA_DOT
