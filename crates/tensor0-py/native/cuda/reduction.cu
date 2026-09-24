#include "execute/reduction.cuh"
#include "arithmetic.cuh"
#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::reduction {

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
ffi::Error Reduction(ffi::Span<const int64_t> words,
                        ffi::Span<const int64_t> coefficient_records,
                        ffi::AnyBuffer source, ffi::BufferR1<ffi::S64> descriptor,
                        ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
                        cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (source.element_type() != Dtype) throw std::invalid_argument("CUDA reduction supports only same-dtype F32/F64/C64/C128 storage");
    if (source.dimensions().size() != 2) throw std::invalid_argument("CUDA reduction source must be rank-two");
    for (const auto dimensions : {source.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("CUDA reduction dimensions must be nonnegative");
      }
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) {
      throw std::invalid_argument("CUDA reduction descriptor operand shape does not match layout");
    }
    // Reconstruct protocol bytes explicitly: the i64 attribute is a bitcast,
    // not a signed-size representation. DecodeReductionLayout validates the
    // stronger reduction contract without CPU execution optimization.
    std::vector<char> bytes;
    bytes.reserve(words.size() * sizeof(uint64_t));
    for (auto word : words) {
      for (unsigned shift = 0; shift < 64; shift += 8)
        bytes.push_back(static_cast<char>(static_cast<uint64_t>(word) >> shift));
    }
    const auto decoded = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
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
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(words.size(), sizeof(int64_t)), output, output_bytes);
    if (coefficient_records.size() != coefficients.size()) throw std::invalid_argument("CUDA reduction coefficient record count does not match operands");
    std::vector<BoundCoefficient> buffers;
    std::vector<std::size_t> parameters(decoded.records.size(), coefficients.size());
    int64_t previous = -1;
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
      const auto record = coefficient_records[i];
      if (record <= previous || record < 0 || static_cast<uint64_t>(record) >= decoded.records.size()) {
        throw std::invalid_argument("CUDA reduction coefficient records must be increasing valid record indices");
      }
      previous = record;
      auto buffer = coefficients.get<ffi::AnyBuffer>(i);
      if (!buffer) throw std::invalid_argument("CUDA reduction coefficient must be a buffer");
      const auto count = CoefficientCount(*buffer, batches);
      VisitCoefficient<T, Real>(buffer->element_type(), [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(buffer->untyped_data(), BufferBytes(count, sizeof(Raw)), output, output_bytes);
      });
      parameters[record] = i;
      buffers.push_back({buffer->untyped_data(), buffer->element_type(), count});
    }
    std::vector<Schedule> schedules;
    std::size_t cursor = 4;
    for (std::size_t i = 0; i < decoded.records.size(); ++i) {
      // The decoder already validated all original extents and role products.
      const auto rank = static_cast<uint64_t>(words[cursor]);
      uint64_t owners = 1, contributions = 1;
      for (uint64_t axis = 0; axis < rank; ++axis) {
        auto& count = words[cursor + 3 + 4 * rank + axis] ? contributions : owners;
        count *= static_cast<uint64_t>(words[cursor + 3 + axis]);
      }
      uint64_t count = 0, total = 0;
      if (!layout::CheckedMultiply(owners, contributions, &count) ||
          !layout::CheckedMultiply(batches, count, &total) ||
          !layout::CheckedMultiply(batches, owners, &total)) {
        throw std::invalid_argument("CUDA reduction logical batch size overflows");
      }
      schedules.push_back({owners, contributions, cursor});
      cursor += 3 + 5 * rank;
    }
    Execute<T, Real>(input, output, decoded.source_size, decoded.output_size, schedules, buffers,
                     parameters, batches, output_bytes, descriptor.typed_data(), stream);
  });
}
}  // namespace tensor0::stride::cuda::reduction

#define TENSOR0_CUDA_REDUCTION(Suffix, Dtype, T, Real) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaReduction##Suffix##V1, (tensor0::stride::cuda::reduction::Reduction<ffi::Dtype, T, Real>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>().RemainingArgs() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaReduction##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaReduction##Suffix##V1); \
}

TENSOR0_CUDA_REDUCTION(F32, F32, float, float)
TENSOR0_CUDA_REDUCTION(F64, F64, double, double)
TENSOR0_CUDA_REDUCTION(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>, float)
TENSOR0_CUDA_REDUCTION(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>, double)
#undef TENSOR0_CUDA_REDUCTION
