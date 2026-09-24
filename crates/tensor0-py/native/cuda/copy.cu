#include <cuda_runtime.h>

#include "../ffi/errors.h"
#include "../ffi/prepared.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda {

// Copy storage bits, including complex components, without arithmetic conversions.
template <int Bytes> struct Storage { unsigned char bytes[Bytes]; };

template <int Bytes>
__global__ void CopyRecord(const Storage<Bytes>* source, Storage<Bytes>* output,
                          const int64_t* record, uint64_t count,
                          uint64_t total, uint64_t source_size,
                          uint64_t output_size) {
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += step) {
    const uint64_t batch = index / count;
    uint64_t logical = index % count;
    const int64_t rank = record[0];
    int64_t source_address = record[1];
    int64_t output_address = record[2];
    for (int64_t axis = rank; axis-- > 0;) {
      const auto coordinate = static_cast<int64_t>(logical % record[3 + axis]);
      logical /= record[3 + axis];
      source_address += coordinate * record[3 + rank + axis];
      output_address += coordinate * record[3 + 2 * rank + axis];
    }
    output[batch * output_size + output_address] = source[batch * source_size + source_address];
  }
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA copy: ") + cudaGetErrorString(status));
  }
}

template <ffi::DataType Dtype, int Bytes>
ffi::Error Copy(ffi::Span<const int64_t> words, ffi::AnyBuffer source,
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
        static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) {
      throw std::invalid_argument("CUDA copy descriptor operand shape does not match layout");
    }
    // The internal lowering supplies the same immutable words as host attribute
    // and XLA-owned device constant. Device buffers are never read on the host.
    const auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
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
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(words.size(), sizeof(int64_t)),
                            output, output_bytes);
    // Validate all records before submitting any writes. Cross-record disjoint
    // destinations remain a producer precondition, as on the CPU path.
    std::vector<uint64_t> counts;
    counts.reserve(decoded.records.size());
    for (const auto& record : decoded.records) {
      const uint64_t count = layout::ElementCount(record);
      std::vector<std::size_t> axes(count == 0 ? 0 : record.shape.size());
      std::iota(axes.begin(), axes.end(), 0);
      layout::ValidateInjectiveView(record, false, std::move(axes));
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, count, &total)) {
        throw std::invalid_argument("CUDA copy logical batch size overflows");
      }
      counts.push_back(count);
    }
    if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
    std::size_t cursor = 4;
    for (std::size_t i = 0; i < decoded.records.size(); ++i) {
      const auto count = counts[i];
      const auto total = batches * count;
      if (total != 0) {
        const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
        CopyRecord<Bytes><<<blocks, 256, 0, stream>>>(input, output,
            descriptor.typed_data() + cursor, count, total, decoded.source_size, decoded.output_size);
        Check(cudaGetLastError());
      }
      cursor += 3 + 3 * decoded.records[i].shape.size();
    }
  });
}

}  // namespace tensor0::stride::cuda

#define TENSOR0_CUDA_COPY(Suffix, Dtype, Bytes) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaCopy##Suffix##V1, (tensor0::stride::cuda::Copy<ffi::Dtype, Bytes>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
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
