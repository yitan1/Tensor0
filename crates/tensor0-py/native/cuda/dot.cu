#include "execute/dot.cuh"
#include "arithmetic.cuh"
#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ffi = xla::ffi;
namespace tensor0::stride::cuda::dot {

template <ffi::DataType Dtype, typename T>
ffi::Error Dot(ffi::Span<const int64_t> words, int64_t conjugate,
               ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::BufferR1<ffi::S64> descriptor,
               ffi::ResultBufferR1<Dtype> result, ffi::ResultBufferR2<Dtype> scratch,
               cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (left.element_type() != Dtype || right.element_type() != Dtype) throw std::invalid_argument("CUDA dot supports only same-dtype F32/F64/C64/C128 inputs and result");
    if (conjugate != 0 && conjugate != 1) throw std::invalid_argument("CUDA dot conjugate_left must be zero or one");
    if (left.dimensions().size() != 2 || right.dimensions().size() != 2) throw std::invalid_argument("CUDA dot inputs must be rank-two");
    for (const auto dimensions : {left.dimensions(), right.dimensions(), result->dimensions(), scratch->dimensions()}) {
      for (auto dimension : dimensions) if (dimension < 0) throw std::invalid_argument("CUDA dot dimensions must be nonnegative");
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) throw std::invalid_argument("CUDA dot descriptor operand shape does not match layout");
    const auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
    const uint64_t batches = result->dimensions()[0];
    if (static_cast<uint64_t>(left.dimensions()[0]) != batches || static_cast<uint64_t>(right.dimensions()[0]) != batches ||
        static_cast<uint64_t>(left.dimensions()[1]) != decoded.source_size || static_cast<uint64_t>(right.dimensions()[1]) != decoded.output_size ||
        static_cast<uint64_t>(scratch->dimensions()[0]) != batches) throw std::invalid_argument("CUDA dot buffer dimensions do not match layout");
    std::vector<uint64_t> counts;
    uint64_t capacity = 0;
    for (const auto& record : decoded.records) {
      const auto count = layout::ElementCount(record);
      uint64_t total;
      if (!layout::CheckedMultiply(count, batches, &total)) throw std::invalid_argument("CUDA dot logical batch size overflows");
      counts.push_back(count);
      capacity = std::max(capacity, count == 0 ? 0 : (count - 1) / kChunk + 1);
    }
    if (static_cast<uint64_t>(scratch->dimensions()[1]) != capacity) throw std::invalid_argument("CUDA dot scratch capacity does not match layout");
    uint64_t left_elements, right_elements, scratch_elements;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &left_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &right_elements) ||
        !layout::CheckedMultiply(batches, capacity, &scratch_elements)) throw std::invalid_argument("CUDA dot batch storage size overflows");
    const auto output_bytes = BufferBytes(batches, sizeof(T));
    const auto scratch_bytes = BufferBytes(scratch_elements, sizeof(T));
    const auto left_bytes = BufferBytes(left_elements, sizeof(T));
    const auto right_bytes = BufferBytes(right_elements, sizeof(T));
    const auto descriptor_bytes = BufferBytes(words.size(), sizeof(int64_t));
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    auto* temporary = reinterpret_cast<T*>(scratch->untyped_data());
    for (const auto target : {std::pair<void*, uint64_t>{output, output_bytes}, {temporary, scratch_bytes}}) {
      ValidateDisjointBuffers(left.untyped_data(), left_bytes, target.first, target.second);
      ValidateDisjointBuffers(right.untyped_data(), right_bytes, target.first, target.second);
      ValidateDisjointBuffers(descriptor.typed_data(), descriptor_bytes, target.first, target.second);
    }
    ValidateDisjointBuffers(output, output_bytes, temporary, scratch_bytes);
    Execute<T>(reinterpret_cast<const T*>(left.untyped_data()),
        reinterpret_cast<const T*>(right.untyped_data()), output, temporary,
        capacity, conjugate != 0, decoded, counts, batches, output_bytes,
        descriptor.typed_data(), stream);
  });
}
}  // namespace tensor0::stride::cuda::dot

#define TENSOR0_CUDA_DOT(Suffix, Dtype, T) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaDot##Suffix##V1, (tensor0::stride::cuda::dot::Dot<ffi::Dtype, T>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout").Attr<int64_t>("conjugate_left") \
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
