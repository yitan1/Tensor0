#pragma once

#include <cuda_runtime.h>
#include <cstdint>

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

}  // namespace tensor0::stride::cuda
