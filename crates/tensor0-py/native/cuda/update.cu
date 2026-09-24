#include <cuda_runtime.h>
#include "arithmetic.cuh"

#include "../ffi/errors.h"
#include "../ffi/prepared.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::update {

using namespace arithmetic;

template <typename T, typename Real, typename Alpha, typename Beta>
__global__ void UpdateRecord(const T* source, const T* base, T* output,
                             const Alpha* alpha, uint64_t alpha_count,
                             const Beta* beta, uint64_t beta_count,
                             const int64_t* record, uint64_t count, uint64_t total,
                             uint64_t source_size, uint64_t output_size) {
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += step) {
    const uint64_t batch = index / count;
    uint64_t logical = index % count;
    const int64_t rank = record[0];
    int64_t source_address = record[1], destination_address = record[2];
    for (int64_t axis = rank; axis-- > 0;) {
      const auto coordinate = static_cast<int64_t>(logical % record[3 + axis]);
      logical /= record[3 + axis];
      source_address += coordinate * record[3 + rank + axis];
      destination_address += coordinate * record[3 + 2 * rank + axis];
    }
    const uint64_t src = batch * source_size + source_address;
    const uint64_t dst = batch * output_size + destination_address;
    const auto a = ReadCoefficient<Real>(alpha, alpha_count, batch);
    const auto b = ReadCoefficient<Real>(beta, beta_count, batch);
    T value;
    if (Zero(a)) {
      if (Zero(b)) value = T{};
      else value = One(b) ? base[dst] : Product(b, base[dst]);
    } else if (Zero(b)) {
      value = One(a) ? source[src] : Product(a, source[src]);
    } else {
      const auto source_term = One(a) ? source[src] : Product(a, source[src]);
      const auto base_term = One(b) ? base[dst] : Product(b, base[dst]);
      value = Sum(source_term, base_term);
    }
    output[dst] = value;
  }
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA update: ") + cudaGetErrorString(status));
  }
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
    // All metadata and overlap validation precedes any device writes.
    VisitCoefficient<T, Real>(alpha.element_type(), [&](auto alpha_type) {
      using Alpha = typename decltype(alpha_type)::type;
      VisitCoefficient<T, Real>(beta.element_type(), [&](auto beta_type) {
        using Beta = typename decltype(beta_type)::type;
        if (output_bytes != 0 && old != output) {
          Check(cudaMemcpyAsync(output, old, output_bytes, cudaMemcpyDeviceToDevice, stream));
        }
        std::size_t cursor = 4;
        for (std::size_t i = 0; i < decoded.records.size(); ++i) {
          const auto total = batches * counts[i];
          if (total != 0) {
            const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
            UpdateRecord<T, Real, Alpha, Beta><<<blocks, 256, 0, stream>>>(
                input, old, output, reinterpret_cast<const Alpha*>(alpha.untyped_data()), alpha_count,
                reinterpret_cast<const Beta*>(beta.untyped_data()), beta_count,
                descriptor.typed_data() + cursor, counts[i], total, decoded.source_size, decoded.output_size);
            Check(cudaGetLastError());
          }
          cursor += 3 + 3 * decoded.records[i].shape.size();
        }
      });
    });
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
