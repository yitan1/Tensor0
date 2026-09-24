#include "arithmetic.cuh"
#include "../ffi/errors.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride::cuda::accumulation {
using namespace arithmetic;

// With owned fibers, map coordinates and zero-stride coordinates are decoded
// separately. The general fallback instead decodes all axes in logical order.
__device__ void Address(const int64_t* record, uint64_t logical, bool fiber,
                        bool general, int64_t& source, int64_t& destination) {
  const int64_t rank = record[0];
  for (int64_t axis = rank; axis-- > 0;) {
    if (!general && (record[3 + 2 * rank + axis] == 0) != fiber) continue;
    const auto coordinate = static_cast<int64_t>(logical % record[3 + axis]);
    logical /= record[3 + axis];
    source += coordinate * record[3 + rank + axis];
    destination += coordinate * record[3 + 2 * rank + axis];
  }
}

template <typename T, typename Real, typename Raw>
__global__ void AccumulateRecord(const T* source, T* output, const Raw* coefficient,
                                 uint64_t coefficient_count, const int64_t* record,
                                 uint64_t owners, uint64_t contributions, uint64_t total,
                                 bool general, uint64_t source_size, uint64_t output_size) {
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;) {
    const uint64_t batch = index / owners;
    const auto factor = coefficient == nullptr ? Raw{1} : ReadCoefficient<Real>(coefficient, coefficient_count, batch);
    if (!Zero(factor)) {
      int64_t map_source = record[1], map_destination = record[2];
      if (!general) Address(record, index % owners, false, false, map_source, map_destination);
      const auto* input = source + batch * source_size;
      auto* result = output + batch * output_size;
      // A fiber starts at the existing output, preserving cross-record grouping.
      T value{};
      if (!general) value = result[map_destination];
      for (uint64_t logical = 0; logical < contributions; ++logical) {
        int64_t src = map_source, dst = map_destination;
        Address(record, logical, true, general, src, dst);
        const T term = One(factor) ? input[src] : Product(factor, input[src]);
        if (general) result[dst] = Sum(result[dst], term);
        else value = Sum(value, term);
      }
      if (!general) result[map_destination] = value;
    }
    // Unlike output storage, contribution counts can approach uint64's limit.
    if (total - index <= step) break;
    index += step;
  }
}

struct Schedule {
  uint64_t owners, contributions;
  bool general;
};

Schedule Classify(const layout::Record& record, uint64_t count) {
  if (count == 0) return {0, 0, false};
  std::vector<std::size_t> map_axes;
  uint64_t owners = 1;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if (record.shape[axis] > 1 && record.destination_strides[axis] != 0) {
      map_axes.push_back(axis);
      // ElementCount has already checked the product of all nonempty extents.
      owners *= record.shape[axis];
    }
  }
  try {
    layout::ValidateInjectiveView(record, false, std::move(map_axes));
    return {owners, count / owners, false};
  } catch (const std::invalid_argument&) {
    // Failure of the sufficient proof is not proof of a collision.
    return {1, count, true};
  }
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA accumulation: ") + cudaGetErrorString(status));
}

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
    std::vector<ffi::AnyBuffer> buffers;
    std::vector<uint64_t> coefficient_counts;
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
      buffers.push_back(*buffer);
      coefficient_counts.push_back(count);
    }
    std::vector<Schedule> schedules;
    for (const auto& record : decoded.records) {
      const uint64_t count = layout::ElementCount(record);
      uint64_t total = 0;
      if (!layout::CheckedMultiply(batches, count, &total)) throw std::invalid_argument("CUDA accumulation logical batch size overflows");
      schedules.push_back(Classify(record, count));
    }
    // DecodeAddressLayout bounds each positive/negative span independently from
    // the offset. Every partial coordinate sum, in any axis order, is within
    // those signed bounds; original-axis device address arithmetic is safe.
    // All validation, including empty-record coefficients, precedes writes.
    if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
    std::size_t cursor = 4;
    for (std::size_t i = 0; i < decoded.records.size(); ++i) {
      const auto schedule = schedules[i];
      const auto total = batches * schedule.owners;
      if (total != 0) {
        const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
        const auto parameter = parameters[i];
        auto launch = [&](auto type) {
          using Raw = typename decltype(type)::type;
          const Raw* coefficient = parameter == buffers.size() ? nullptr : reinterpret_cast<const Raw*>(buffers[parameter].untyped_data());
          const uint64_t count = parameter == buffers.size() ? 1 : coefficient_counts[parameter];
          AccumulateRecord<T, Real, Raw><<<blocks, 256, 0, stream>>>(input, output, coefficient, count,
              descriptor.typed_data() + cursor, schedule.owners, schedule.contributions, total,
              schedule.general, decoded.source_size, decoded.output_size);
          Check(cudaGetLastError());
        };
        if (parameter == buffers.size()) launch(std::type_identity<Real>{});
        else VisitCoefficient<T, Real>(buffers[parameter].element_type(), launch);
      }
      cursor += 3 + 3 * decoded.records[i].shape.size();
    }
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
