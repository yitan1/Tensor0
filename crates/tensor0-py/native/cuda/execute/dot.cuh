#pragma once

#include "../kernels/dot.cuh"
#include "../../layout/descriptor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor0::stride::cuda::dot {

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA dot: ") + cudaGetErrorString(status));
}

template <typename T>
void Execute(const T* left, const T* right, T* output, T* temporary,
    uint64_t capacity, bool conjugate,
    const descriptor::DecodedLayout& decoded,
    const std::vector<uint64_t>& counts, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  // All metadata, sizes and aliases are validated before either result is written.
  if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
  std::size_t cursor = 4;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const auto count = counts[i];
    if (count != 0 && batches != 0) {
      const uint64_t chunks = (count - 1) / kChunk + 1;
      const uint64_t tasks = batches * chunks;
      Partials<T><<<static_cast<unsigned>(std::min<uint64_t>(tasks, 65535)), kThreads, 0, stream>>>(
          left, right,
          temporary, device_descriptor + cursor, count, chunks, capacity, tasks,
          decoded.source_size, decoded.output_size, conjugate != 0);
      Check(cudaGetLastError());
      Finish<T><<<static_cast<unsigned>(std::min<uint64_t>(batches, 65535)), kThreads, 0, stream>>>(temporary, output, chunks, capacity, batches);
      Check(cudaGetLastError());
    }
    cursor += 3 + 3 * decoded.records[i].shape.size();
  }
}

}  // namespace tensor0::stride::cuda::dot
