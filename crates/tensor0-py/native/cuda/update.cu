#include "execute/update.cuh"
#include <cuda_runtime.h>
#include "arithmetic.cuh"

#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"
#include "../layout/owner_fiber.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::update {

namespace {
std::atomic<uint64_t> update_prepared_created_count{0};
std::atomic<uint64_t> update_prepared_destroyed_count{0};
}  // namespace

struct UpdatePreparedState {
  static ffi::TypeId id;

  UpdatePreparedState(descriptor::DecodedLayout value, std::vector<uint64_t> sizes,
                      std::vector<bool> identities, std::vector<layout::OwnerFiberSchedule> tasks,
                      std::size_t words, bool covers_output)
      : decoded(std::move(value)), counts(std::move(sizes)),
        identity(std::move(identities)), schedules(std::move(tasks)), word_count(words),
        full_coverage(covers_output) {
    update_prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }
  UpdatePreparedState(const UpdatePreparedState&) = delete;
  UpdatePreparedState& operator=(const UpdatePreparedState&) = delete;
  ~UpdatePreparedState() {
    update_prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const descriptor::DecodedLayout decoded;
  const std::vector<uint64_t> counts;
  const std::vector<bool> identity;
  const std::vector<layout::OwnerFiberSchedule> schedules;
  const std::size_t word_count;
  const bool full_coverage;
};

ffi::TypeId UpdatePreparedState::id = {};
const ffi::TypeInfo kUpdatePreparedTypeInfo = ffi::MakeTypeInfo<UpdatePreparedState>();

ffi::ErrorOr<std::unique_ptr<UpdatePreparedState>> InstantiateUpdate(
    ffi::Span<const int64_t> words) {
  std::unique_ptr<UpdatePreparedState> state;
  const auto error = ContainErrors([&] {
    // Preserve all axes and records; device descriptor words retain their offsets.
    auto decoded = descriptor::DecodeLayout(words.begin(), words.size());
    std::vector<uint64_t> counts;
    std::vector<bool> identity;
    counts.reserve(decoded.records.size());
    identity.reserve(decoded.records.size());
    for (const auto& record : decoded.records) {
      const auto count = layout::ElementCount(record);
      counts.push_back(count);
      identity.push_back(layout::HasIdenticalAddresses(record));
    }
    auto packed = layout::PackOwnerFiber(decoded, 1);
    const bool full_coverage = FullyCoversOutput(decoded);
    state = std::make_unique<UpdatePreparedState>(
        std::move(decoded), std::move(counts), std::move(identity),
        std::move(packed.schedules), packed.words.size(), full_coverage);
  });
  if (error.failure()) return ffi::Unexpected(error);
  return std::move(state);
}

uint64_t CoefficientCount(ffi::AnyBuffer buffer, uint64_t batches) {
  const auto dimensions = buffer.dimensions();
  if (dimensions.size() == 0) return 1;
  if (dimensions.size() != 1 || dimensions[0] < 0) {
    throw std::invalid_argument("CUDA update coefficients must be scalar or rank-one");
  }
  const uint64_t count = dimensions[0];
  if (count != 1 && count != batches) {
    throw std::invalid_argument("CUDA update coefficients must be shared or one value per batch");
  }
  return count;
}

template <ffi::DataType Dtype, typename T, typename Real>
ffi::Error Update(const UpdatePreparedState* prepared, ffi::AnyBuffer source,
                  ffi::BufferR2<Dtype> base, ffi::AnyBuffer alpha, ffi::AnyBuffer beta,
                  ffi::BufferR1<ffi::S64> descriptor,
                  ffi::ResultBufferR2<Dtype> result, cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (source.element_type() != Dtype) {
      throw std::invalid_argument("CUDA update supports only same-dtype F32/F64/C64/C128 storage");
    }
    if (source.dimensions().size() != 2) {
      throw std::invalid_argument("CUDA update source storage must be rank-two");
    }
    for (const auto dimensions : {source.dimensions(), base.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("CUDA update dimensions must be nonnegative");
      }
    }
    if (descriptor.dimensions()[0] < 0 ||
        static_cast<uint64_t>(descriptor.dimensions()[0]) != prepared->word_count) {
      throw std::invalid_argument("CUDA update descriptor operand shape does not match layout");
    }
    // The lowering supplies identical immutable host words and device constant.
    const auto& decoded = prepared->decoded;
    if (source.dimensions()[0] != result->dimensions()[0] ||
        base.dimensions()[0] != result->dimensions()[0] ||
        base.dimensions()[1] != result->dimensions()[1] ||
        static_cast<uint64_t>(source.dimensions()[1]) != decoded.source_size ||
        static_cast<uint64_t>(result->dimensions()[1]) != decoded.output_size) {
      throw std::invalid_argument("CUDA update buffer dimensions do not match layout");
    }
    const uint64_t batches = result->dimensions()[0];
    uint64_t source_elements = 0, output_elements = 0;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &source_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &output_elements)) {
      throw std::invalid_argument("CUDA update batch storage size overflows");
    }
    const auto source_bytes = BufferBytes(source_elements, sizeof(T));
    const auto output_bytes = BufferBytes(output_elements, sizeof(T));
    const auto* input = reinterpret_cast<const T*>(source.untyped_data());
    const auto* old = reinterpret_cast<const T*>(base.untyped_data());
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    const auto alpha_count = CoefficientCount(alpha, batches);
    const auto beta_count = CoefficientCount(beta, batches);
    for (const auto& coefficient : {alpha, beta}) {
      const uint64_t count = CoefficientCount(coefficient, batches);
      arithmetic::VisitCoefficient<T, Real>(coefficient.element_type(), [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(coefficient.untyped_data(), BufferBytes(count, sizeof(Raw)), output, output_bytes);
      });
    }
    if (old != output) ValidateDisjointBuffers(old, output_bytes, output, output_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(prepared->word_count, sizeof(int64_t)), output, output_bytes);
    const bool source_alias = source_bytes != 0 && output_bytes != 0 && input == output;
    if (source_alias) {
      if (decoded.source_size != decoded.output_size || old != output) {
        throw std::invalid_argument("unsupported CUDA update source/result alias: requires identical source/base/result storage");
      }
    } else {
      ValidateDisjointBuffers(input, source_bytes, output, output_bytes);
    }
    for (std::size_t i = 0; i < prepared->counts.size(); ++i) {
      if (source_alias && !prepared->identity[i]) {
        throw std::invalid_argument("CUDA update source/result alias requires identical per-element addresses");
      }
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, prepared->counts[i], &total)) {
        throw std::invalid_argument("CUDA update logical batch size overflows");
      }
    }
    Execute<T, Real>(input, old, output,
        {alpha.untyped_data(), alpha.element_type(), alpha_count},
        {beta.untyped_data(), beta.element_type(), beta_count},
        decoded, prepared->schedules, batches, output_bytes, prepared->full_coverage,
        descriptor.typed_data(), stream);
  });
}

}  // namespace tensor0::stride::cuda::update

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideCudaUpdateInstantiateV1, tensor0::stride::cuda::update::InstantiateUpdate,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout"));

extern "C" void* Tensor0StrideCudaUpdateInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideCudaUpdateInstantiateV1);
}

extern "C" void* Tensor0StrideCudaUpdatePreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::cuda::update::UpdatePreparedState::id);
}

extern "C" const void* Tensor0StrideCudaUpdatePreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::cuda::update::kUpdatePreparedTypeInfo);
}

extern "C" uint64_t Tensor0StrideCudaUpdatePreparedCreatedCount() {
  return tensor0::stride::cuda::update::update_prepared_created_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideCudaUpdatePreparedDestroyedCount() {
  return tensor0::stride::cuda::update::update_prepared_destroyed_count.load(std::memory_order_relaxed);
}

#define TENSOR0_CUDA_UPDATE(Suffix, Dtype, T, Real) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaUpdate##Suffix##V1, (tensor0::stride::cuda::update::Update<ffi::Dtype, T, Real>), \
    ffi::Ffi::BindExecute().Ctx<ffi::State<tensor0::stride::cuda::update::UpdatePreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR2<ffi::Dtype>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaUpdate##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaUpdate##Suffix##V1); \
}

TENSOR0_CUDA_UPDATE(F32, F32, float, float)
TENSOR0_CUDA_UPDATE(F64, F64, double, double)
TENSOR0_CUDA_UPDATE(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>, float)
TENSOR0_CUDA_UPDATE(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>, double)

#undef TENSOR0_CUDA_UPDATE
