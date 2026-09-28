#include "execute/copy.cuh"
#include <cuda_runtime.h>

#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"
#include "../layout/owner_fiber.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda {

namespace {
std::atomic<uint64_t> copy_prepared_created_count{0};
std::atomic<uint64_t> copy_prepared_destroyed_count{0};
}  // namespace

struct CopyPreparedState {
  static ffi::TypeId id;

  CopyPreparedState(descriptor::DecodedLayout value, std::vector<uint64_t> sizes,
                    std::vector<layout::OwnerFiberSchedule> tasks, std::size_t words,
                    bool covers_output)
      : decoded(std::move(value)), counts(std::move(sizes)),
        schedules(std::move(tasks)), word_count(words), full_coverage(covers_output) {
    copy_prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }
  CopyPreparedState(const CopyPreparedState&) = delete;
  CopyPreparedState& operator=(const CopyPreparedState&) = delete;
  ~CopyPreparedState() {
    copy_prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const descriptor::DecodedLayout decoded;
  const std::vector<uint64_t> counts;
  const std::vector<layout::OwnerFiberSchedule> schedules;
  const std::size_t word_count;
  const bool full_coverage;
};

ffi::TypeId CopyPreparedState::id = {};
const ffi::TypeInfo kCopyPreparedTypeInfo = ffi::MakeTypeInfo<CopyPreparedState>();

ffi::ErrorOr<std::unique_ptr<CopyPreparedState>> InstantiateCopy(
    ffi::Span<const int64_t> words) {
  std::unique_ptr<CopyPreparedState> state;
  const auto error = ContainErrors([&] {
    // Final descriptor words are authoritative: preserve every axis and record.
    auto decoded = descriptor::DecodeLayout(words.begin(), words.size());
    std::vector<uint64_t> counts;
    counts.reserve(decoded.records.size());
    for (const auto& record : decoded.records) {
      counts.push_back(layout::ElementCount(record));
    }
    auto packed = layout::PackOwnerFiber(decoded, 0);
    const bool full_coverage = FullyCoversOutput(decoded);
    state = std::make_unique<CopyPreparedState>(
        std::move(decoded), std::move(counts), std::move(packed.schedules), packed.words.size(),
        full_coverage);
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

template <ffi::DataType Dtype, int Bytes>
ffi::Error Copy(const CopyPreparedState* prepared, ffi::AnyBuffer source,
                ffi::BufferR1<ffi::S64> descriptor,
                ffi::ResultBufferR2<Dtype> result, cudaStream_t stream) {
  return ContainErrors([&] {
    if (source.element_type() != Dtype) {
      throw std::invalid_argument("CUDA copy supports only same-dtype F32/F64/C64/C128; conversion is unsupported");
    }
    if (source.dimensions().size() != 2) {
      throw std::invalid_argument("CUDA copy storage must be rank-two");
    }
    for (const auto dimensions : {source.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("CUDA copy dimensions must be nonnegative");
      }
    }
    if (descriptor.dimensions()[0] < 0 ||
        static_cast<uint64_t>(descriptor.dimensions()[0]) != prepared->word_count) {
      throw std::invalid_argument("CUDA copy descriptor operand shape does not match layout");
    }
    // The internal lowering supplies the same immutable words as host attribute
    // and XLA-owned device constant. Device buffers are never read on the host.
    const auto& decoded = prepared->decoded;
    if (source.dimensions()[0] != result->dimensions()[0] ||
        static_cast<uint64_t>(source.dimensions()[1]) != decoded.source_size ||
        static_cast<uint64_t>(result->dimensions()[1]) != decoded.output_size) {
      throw std::invalid_argument("CUDA copy buffer dimensions do not match layout");
    }
    const uint64_t batches = source.dimensions()[0];
    uint64_t source_elements = 0, output_elements = 0;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &source_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &output_elements)) {
      throw std::invalid_argument("CUDA copy batch storage size overflows");
    }
    const auto source_bytes = BufferBytes(source_elements, Bytes);
    const auto output_bytes = BufferBytes(output_elements, Bytes);
    const auto* input = reinterpret_cast<const Storage<Bytes>*>(source.untyped_data());
    auto* output = reinterpret_cast<Storage<Bytes>*>(result->untyped_data());
    ValidateDisjointBuffers(input, source_bytes, output, output_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(prepared->word_count, sizeof(int64_t)),
                            output, output_bytes);
    // Validate runtime batch sizes before submitting any writes. Static map
    // validation and element counts belong to instantiate. Cross-record disjoint
    // destinations remain a producer precondition, as on the CPU path.
    for (const auto count : prepared->counts) {
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, count, &total)) {
        throw std::invalid_argument("CUDA copy logical batch size overflows");
      }
    }
    ExecuteCopy<Bytes>(input, output, decoded, prepared->schedules, batches, output_bytes,
                       prepared->full_coverage, descriptor.typed_data(), stream);
  });
}

}  // namespace tensor0::stride::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideCudaCopyInstantiateV1, tensor0::stride::cuda::InstantiateCopy,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout"));

extern "C" void* Tensor0StrideCudaCopyInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideCudaCopyInstantiateV1);
}

extern "C" void* Tensor0StrideCudaCopyPreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::cuda::CopyPreparedState::id);
}

extern "C" const void* Tensor0StrideCudaCopyPreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::cuda::kCopyPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StrideCudaCopyPreparedCreatedCount() {
  return tensor0::stride::cuda::copy_prepared_created_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideCudaCopyPreparedDestroyedCount() {
  return tensor0::stride::cuda::copy_prepared_destroyed_count.load(std::memory_order_relaxed);
}

#define TENSOR0_CUDA_COPY(Suffix, Dtype, Bytes) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaCopy##Suffix##V1, (tensor0::stride::cuda::Copy<ffi::Dtype, Bytes>), \
    ffi::Ffi::BindExecute().Ctx<ffi::State<tensor0::stride::cuda::CopyPreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaCopy##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaCopy##Suffix##V1); \
}

TENSOR0_CUDA_COPY(F32, F32, 4)
TENSOR0_CUDA_COPY(F64, F64, 8)
TENSOR0_CUDA_COPY(C64, C64, 8)
TENSOR0_CUDA_COPY(C128, C128, 16)

#undef TENSOR0_CUDA_COPY
