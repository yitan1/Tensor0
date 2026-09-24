#pragma once

#include "../arithmetic.cuh"

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

}  // namespace tensor0::stride::cuda::accumulation
