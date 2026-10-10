#include "execute/sum.cuh"
#include "arithmetic.cuh"
#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::reduction {

using Schedule = layout::OwnerFiberSchedule;

namespace {
std::atomic<uint64_t> reduction_prepared_created_count{0};
std::atomic<uint64_t> reduction_prepared_destroyed_count{0};
}  // namespace

struct ReductionPreparedState {
  static ffi::TypeId id;

  ReductionPreparedState(descriptor::DecodedReductionLayout value, std::vector<Schedule> plans,
                         std::vector<uint64_t> sizes, std::vector<std::size_t> indices,
                         std::size_t coefficients, std::size_t words, uint64_t scratch_capacity)
      : decoded(std::move(value)), schedules(std::move(plans)), counts(std::move(sizes)),
        parameters(std::move(indices)),
        coefficient_count(coefficients), word_count(words), capacity(scratch_capacity) {
    reduction_prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }
  ReductionPreparedState(const ReductionPreparedState&) = delete;
  ReductionPreparedState& operator=(const ReductionPreparedState&) = delete;
  ~ReductionPreparedState() {
    reduction_prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const descriptor::DecodedReductionLayout decoded;
  const std::vector<Schedule> schedules;
  const std::vector<uint64_t> counts;
  const std::vector<std::size_t> parameters;
  const std::size_t coefficient_count;
  const std::size_t word_count;
  const uint64_t capacity;
};

ffi::TypeId ReductionPreparedState::id = {};
const ffi::TypeInfo kReductionPreparedTypeInfo = ffi::MakeTypeInfo<ReductionPreparedState>();

ffi::ErrorOr<std::unique_ptr<ReductionPreparedState>> InstantiateReduction(
    ffi::Span<const int64_t> words, ffi::Span<const int64_t> coefficient_records) {
  std::unique_ptr<ReductionPreparedState> state;
  const auto error = ContainErrors([&] {
    // Preserve full decoder validation (including prefix product overflow) before
    // checking coefficient indices or deriving any schedule from the wire words.
    std::vector<char> bytes;
    bytes.reserve(words.size() * sizeof(uint64_t));
    for (auto word : words) {
      for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<char>(static_cast<uint64_t>(word) >> shift));
    }
    auto decoded = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
    std::vector<std::size_t> parameters(decoded.records.size(), coefficient_records.size());
    int64_t previous = -1;
    for (std::size_t i = 0; i < coefficient_records.size(); ++i) {
      const auto record = coefficient_records[i];
      if (record <= previous || record < 0 || static_cast<uint64_t>(record) >= decoded.records.size()) {
        throw std::invalid_argument("CUDA reduction coefficient records must be increasing valid record indices");
      }
      previous = record;
      parameters[record] = i;
    }
    auto packed = layout::PackOwnerFiber(decoded);
    std::vector<uint64_t> counts;
    counts.reserve(decoded.records.size());
    for (const auto& record : decoded.records) counts.push_back(layout::ElementCount(record.layout));
    const auto capacity = layout::SumScratchCapacity(packed.schedules);
    state = std::make_unique<ReductionPreparedState>(std::move(decoded), std::move(packed.schedules), std::move(counts),
        std::move(parameters), coefficient_records.size(), packed.words.size(), capacity);
  });
  if (error.failure()) return ffi::Unexpected(error);
  return std::move(state);
}

uint64_t CoefficientCount(ffi::AnyBuffer buffer, uint64_t batches) {
  const auto dimensions = buffer.dimensions();
  if (dimensions.size() == 0) return 1;
  if (dimensions.size() != 1 || dimensions[0] < 0 ||
      (dimensions[0] != 1 && static_cast<uint64_t>(dimensions[0]) != batches)) {
    throw std::invalid_argument("CUDA reduction coefficients must be scalar, length-one, or match batch count");
  }
  return dimensions[0];
}

template <ffi::DataType Dtype, typename T, typename Real>
ffi::Error Reduction(const ReductionPreparedState* prepared,
                        ffi::AnyBuffer source, ffi::BufferR1<ffi::S64> descriptor,
                        ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
                        ffi::ResultBufferR2<Dtype> scratch, cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (source.element_type() != Dtype) throw std::invalid_argument("CUDA reduction supports only same-dtype F32/F64/C64/C128 storage");
    if (source.dimensions().size() != 2) throw std::invalid_argument("CUDA reduction source must be rank-two");
    for (const auto dimensions : {source.dimensions(), result->dimensions(), scratch->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("CUDA reduction dimensions must be nonnegative");
      }
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != prepared->word_count) {
      throw std::invalid_argument("CUDA reduction descriptor operand shape does not match layout");
    }
    const auto& decoded = prepared->decoded;
    if (source.dimensions()[0] != result->dimensions()[0] ||
        static_cast<uint64_t>(source.dimensions()[1]) != decoded.source_size ||
        static_cast<uint64_t>(result->dimensions()[1]) != decoded.output_size) {
      throw std::invalid_argument("CUDA reduction buffer dimensions do not match layout");
    }
    const uint64_t batches = result->dimensions()[0];
    uint64_t source_elements = 0, output_elements = 0;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &source_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &output_elements)) {
      throw std::invalid_argument("CUDA reduction batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(output_elements, sizeof(T));
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    const auto* input = reinterpret_cast<const T*>(source.untyped_data());
    ValidateDisjointBuffers(input, BufferBytes(source_elements, sizeof(T)), output, output_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(prepared->word_count, sizeof(int64_t)), output, output_bytes);
    if (prepared->coefficient_count != coefficients.size()) throw std::invalid_argument("CUDA reduction coefficient record count does not match operands");
    std::vector<BoundCoefficient> buffers;
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
      auto buffer = coefficients.get<ffi::AnyBuffer>(i);
      if (!buffer) throw std::invalid_argument("CUDA reduction coefficient must be a buffer");
      const auto count = CoefficientCount(*buffer, batches);
      arithmetic::VisitCoefficient<T, Real>(buffer->element_type(), [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(buffer->untyped_data(), BufferBytes(count, sizeof(Raw)), output, output_bytes);
      });
      buffers.push_back({buffer->untyped_data(), buffer->element_type(), count});
    }
    for (std::size_t i = 0; i < prepared->schedules.size(); ++i) {
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, prepared->counts[i], &total) ||
          !layout::CheckedMultiply(batches, prepared->schedules[i].owners, &total)) {
        throw std::invalid_argument("CUDA reduction logical batch size overflows");
      }
    }
    if (static_cast<uint64_t>(scratch->dimensions()[0]) != batches ||
        static_cast<uint64_t>(scratch->dimensions()[1]) != prepared->capacity)
      throw std::invalid_argument("CUDA reduction scratch capacity does not match layout");
    uint64_t scratch_elements = 0;
    if (!layout::CheckedMultiply(batches, prepared->capacity, &scratch_elements))
      throw std::invalid_argument("CUDA reduction batch storage size overflows");
    const auto scratch_bytes = BufferBytes(scratch_elements, sizeof(T));
    auto* temporary = reinterpret_cast<T*>(scratch->untyped_data());
    ValidateDisjointBuffers(input, BufferBytes(source_elements, sizeof(T)), temporary, scratch_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(prepared->word_count, sizeof(int64_t)), temporary, scratch_bytes);
    ValidateDisjointBuffers(output, output_bytes, temporary, scratch_bytes);
    for (const auto& buffer : buffers) {
      arithmetic::VisitCoefficient<T, Real>(buffer.dtype, [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(buffer.data, BufferBytes(buffer.count, sizeof(Raw)), temporary, scratch_bytes);
      });
    }
    sum::Execute<T, Real>(input, output, temporary, prepared->capacity,
                          decoded.source_size, decoded.output_size, prepared->schedules, buffers,
                          prepared->parameters, batches, output_bytes, descriptor.typed_data(), stream, "reduction");
  });
}
}  // namespace tensor0::stride::cuda::reduction

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideCudaReductionInstantiateV1, tensor0::stride::cuda::reduction::InstantiateReduction,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<ffi::Span<const int64_t>>("coefficient_records"));

extern "C" void* Tensor0StrideCudaReductionInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideCudaReductionInstantiateV1);
}
extern "C" void* Tensor0StrideCudaReductionPreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::cuda::reduction::ReductionPreparedState::id);
}
extern "C" const void* Tensor0StrideCudaReductionPreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::cuda::reduction::kReductionPreparedTypeInfo);
}
extern "C" uint64_t Tensor0StrideCudaReductionPreparedCreatedCount() {
  return tensor0::stride::cuda::reduction::reduction_prepared_created_count.load(std::memory_order_relaxed);
}
extern "C" uint64_t Tensor0StrideCudaReductionPreparedDestroyedCount() {
  return tensor0::stride::cuda::reduction::reduction_prepared_destroyed_count.load(std::memory_order_relaxed);
}

#define TENSOR0_CUDA_REDUCTION(Suffix, Dtype, T, Real) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaReduction##Suffix##V1, (tensor0::stride::cuda::reduction::Reduction<ffi::Dtype, T, Real>), \
    ffi::Ffi::BindExecute().Ctx<ffi::State<tensor0::stride::cuda::reduction::ReductionPreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>().RemainingArgs() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaReduction##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaReduction##Suffix##V1); \
}

TENSOR0_CUDA_REDUCTION(F32, F32, float, float)
TENSOR0_CUDA_REDUCTION(F64, F64, double, double)
TENSOR0_CUDA_REDUCTION(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>, float)
TENSOR0_CUDA_REDUCTION(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>, double)
#undef TENSOR0_CUDA_REDUCTION
