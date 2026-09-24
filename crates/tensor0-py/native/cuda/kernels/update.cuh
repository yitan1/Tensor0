#pragma once

#include "../arithmetic.cuh"

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

}  // namespace tensor0::stride::cuda::update
