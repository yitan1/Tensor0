#pragma once

#include "../arithmetic.cuh"

namespace tensor0::stride::cuda::reduction {
using namespace arithmetic;

// Device metadata is the original unsigned reduction protocol, not the
// optimized host execution record. Axis flags, not output strides, define fibers.
__device__ void Address(const uint64_t* record, uint64_t logical, bool fiber,
                        int64_t& source, int64_t& destination) {
  const uint64_t rank = record[0];
  for (uint64_t axis = rank; axis-- > 0;) {
    if ((record[3 + 4 * rank + axis] != 0) != fiber) continue;
    const uint64_t coordinate = logical % record[3 + axis];
    logical /= record[3 + axis];
    const auto source_stride = static_cast<int64_t>(record[3 + rank + axis]);
    const auto destination_stride = static_cast<int64_t>(record[3 + 3 * rank + axis]);
    // Zero strides allow full uint64 extents without signed coordinate casts.
    if (source_stride != 0) source += static_cast<int64_t>(coordinate) * source_stride;
    if (!fiber && destination_stride != 0) destination += static_cast<int64_t>(coordinate) * destination_stride;
  }
}

template <typename T, typename Real, typename Raw>
__global__ void AccumulateRecord(const T* source, T* output, const Raw* coefficient,
                                 uint64_t coefficient_count, const uint64_t* record,
                                 uint64_t owners, uint64_t contributions, uint64_t total,
                                 uint64_t source_size, uint64_t output_size) {
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;) {
    const uint64_t batch = index / owners;
    const auto factor = coefficient == nullptr ? Raw{1} : ReadCoefficient<Real>(coefficient, coefficient_count, batch);
    if (!Zero(factor)) {
      int64_t map_source = record[1], map_destination = record[2];
      Address(record, index % owners, false, map_source, map_destination);
      const auto* input = source + batch * source_size;
      auto* result = output + batch * output_size;
      // A fiber starts at the existing output, preserving cross-record grouping.
      T value = result[map_destination];
      for (uint64_t logical = 0; logical < contributions; ++logical) {
        int64_t src = map_source, dst = map_destination;
        Address(record, logical, true, src, dst);
        const T term = One(factor) ? input[src] : Product(factor, input[src]);
        value = Sum(value, term);
      }
      result[map_destination] = value;
    }
    // Unlike output storage, contribution counts can approach uint64's limit.
    if (total - index <= step) break;
    index += step;
  }
}

}  // namespace tensor0::stride::cuda::reduction
