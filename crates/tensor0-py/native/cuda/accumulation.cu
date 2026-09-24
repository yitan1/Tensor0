#include "execute/accumulation.cuh"
#include "arithmetic.cuh"
#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::accumulation {

uint64_t CoefficientCount(ffi::AnyBuffer buffer, uint64_t batches) {
  const auto dimensions = buffer.dimensions();
  if (dimensions.size() == 0) return 1;
  if (dimensions.size() != 1 || dimensions[0] < 0 ||
      (dimensions[0] != 1 && static_cast<uint64_t>(dimensions[0]) != batches)) {
    throw std::invalid_argument("CUDA accumulation coefficients must be scalar, length-one, or match batch count");
  }
  return dimensions[0];
}

template <ffi::DataType Dtype, typename T, typename Real>
ffi::Error Accumulation(ffi::Span<const int64_t> words,
                        ffi::Span<const int64_t> coefficient_records,
                        ffi::AnyBuffer source, ffi::BufferR1<ffi::S64> descriptor,
                        ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
                        cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (source.element_type() != Dtype) throw std::invalid_argument("CUDA accumulation supports only same-dtype F32/F64/C64/C128 storage");
    if (source.dimensions().size() != 2) throw std::invalid_argument("CUDA accumulation source must be rank-two");
    for (const auto dimensions : {source.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("CUDA accumulation dimensions must be nonnegative");
      }
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) {
      throw std::invalid_argument("CUDA accumulation descriptor operand shape does not match layout");
    }
    const auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
    if (source.dimensions()[0] != result->dimensions()[0] ||
        static_cast<uint64_t>(source.dimensions()[1]) != decoded.source_size ||
        static_cast<uint64_t>(result->dimensions()[1]) != decoded.output_size) {
      throw std::invalid_argument("CUDA accumulation buffer dimensions do not match layout");
    }
    const uint64_t batches = result->dimensions()[0];
    uint64_t source_elements = 0, output_elements = 0;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &source_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &output_elements)) {
      throw std::invalid_argument("CUDA accumulation batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(output_elements, sizeof(T));
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    const auto* input = reinterpret_cast<const T*>(source.untyped_data());
    ValidateDisjointBuffers(input, BufferBytes(source_elements, sizeof(T)), output, output_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(words.size(), sizeof(int64_t)), output, output_bytes);
    if (coefficient_records.size() != coefficients.size()) throw std::invalid_argument("CUDA accumulation coefficient record count does not match operands");
    std::vector<BoundCoefficient> buffers;
    std::vector<std::size_t> parameters(decoded.records.size(), coefficients.size());
    int64_t previous = -1;
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
      const auto record = coefficient_records[i];
      if (record <= previous || record < 0 || static_cast<uint64_t>(record) >= decoded.records.size()) {
        throw std::invalid_argument("CUDA accumulation coefficient records must be increasing valid record indices");
      }
      previous = record;
      auto buffer = coefficients.get<ffi::AnyBuffer>(i);
      if (!buffer) throw std::invalid_argument("CUDA accumulation coefficient must be a buffer");
      const auto count = CoefficientCount(*buffer, batches);
      VisitCoefficient<T, Real>(buffer->element_type(), [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(buffer->untyped_data(), BufferBytes(count, sizeof(Raw)), output, output_bytes);
      });
      parameters[record] = i;
      buffers.push_back({buffer->untyped_data(), buffer->element_type(), count});
    }
    std::vector<Schedule> schedules;
    for (const auto& record : decoded.records) {
      const uint64_t count = layout::ElementCount(record);
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, count, &total)) throw std::invalid_argument("CUDA accumulation logical batch size overflows");
      schedules.push_back(Classify(record, count));
    }
    Execute<T, Real>(input, output, decoded, schedules, buffers,
                     parameters, batches, output_bytes, descriptor.typed_data(), stream);
  });
}
}  // namespace tensor0::stride::cuda::accumulation

#define TENSOR0_CUDA_ACCUMULATION(Suffix, Dtype, T, Real) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaAccumulation##Suffix##V1, (tensor0::stride::cuda::accumulation::Accumulation<ffi::Dtype, T, Real>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>().RemainingArgs() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaAccumulation##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaAccumulation##Suffix##V1); \
}

TENSOR0_CUDA_ACCUMULATION(F32, F32, float, float)
TENSOR0_CUDA_ACCUMULATION(F64, F64, double, double)
TENSOR0_CUDA_ACCUMULATION(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>, float)
TENSOR0_CUDA_ACCUMULATION(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>, double)
#undef TENSOR0_CUDA_ACCUMULATION
