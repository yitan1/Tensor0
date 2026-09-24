#include "execute/update.cuh"
#include <cuda_runtime.h>
#include "arithmetic.cuh"

#include "../ffi/errors.h"
#include "../ffi/buffers.h"
#include "../layout/descriptor.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::update {

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
ffi::Error Update(ffi::Span<const int64_t> words, ffi::AnyBuffer source,
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
        static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) {
      throw std::invalid_argument("CUDA update descriptor operand shape does not match layout");
    }
    // The lowering supplies identical words as host attribute and device constant.
    const auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
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
      VisitCoefficient<T, Real>(coefficient.element_type(), [&](auto type) {
        using Raw = typename decltype(type)::type;
        ValidateDisjointBuffers(coefficient.untyped_data(), BufferBytes(count, sizeof(Raw)), output, output_bytes);
      });
    }
    if (old != output) ValidateDisjointBuffers(old, output_bytes, output, output_bytes);
    ValidateDisjointBuffers(descriptor.typed_data(), BufferBytes(words.size(), sizeof(int64_t)), output, output_bytes);
    const bool source_alias = source_bytes != 0 && output_bytes != 0 && input == output;
    if (source_alias) {
      if (decoded.source_size != decoded.output_size || old != output) {
        throw std::invalid_argument("unsupported CUDA update source/result alias: requires identical source/base/result storage");
      }
    } else {
      ValidateDisjointBuffers(input, source_bytes, output, output_bytes);
    }
    std::vector<uint64_t> counts;
    counts.reserve(decoded.records.size());
    for (const auto& record : decoded.records) {
      const auto count = layout::ElementCount(record);
      std::vector<std::size_t> axes(count == 0 ? 0 : record.shape.size());
      std::iota(axes.begin(), axes.end(), 0);
      layout::ValidateInjectiveView(record, false, std::move(axes));
      if (source_alias) {
        bool identity = record.source_offset == record.destination_offset;
        for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
          // CPU normalization removes singleton axes only from nonempty records.
          if (count == 0 || record.shape[axis] > 1) {
            identity = identity && record.source_strides[axis] == record.destination_strides[axis];
          }
        }
        if (!identity) throw std::invalid_argument("CUDA update source/result alias requires identical per-element addresses");
      }
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, count, &total)) {
        throw std::invalid_argument("CUDA update logical batch size overflows");
      }
      counts.push_back(count);
    }
    Execute<T, Real>(input, old, output,
        {alpha.untyped_data(), alpha.element_type(), alpha_count},
        {beta.untyped_data(), beta.element_type(), beta_count},
        decoded, counts, batches, output_bytes, descriptor.typed_data(), stream);
  });
}

}  // namespace tensor0::stride::cuda::update

#define TENSOR0_CUDA_UPDATE(Suffix, Dtype, T, Real) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaUpdate##Suffix##V1, (tensor0::stride::cuda::update::Update<ffi::Dtype, T, Real>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR2<ffi::Dtype>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaUpdate##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaUpdate##Suffix##V1); \
}

TENSOR0_CUDA_UPDATE(F32, F32, float, float)
TENSOR0_CUDA_UPDATE(F64, F64, double, double)
TENSOR0_CUDA_UPDATE(C64, C64, tensor0::stride::cuda::update::Complex<float>, float)
TENSOR0_CUDA_UPDATE(C128, C128, tensor0::stride::cuda::update::Complex<double>, double)

#undef TENSOR0_CUDA_UPDATE
