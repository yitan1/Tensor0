#pragma once

#include "../kernels/copy.cuh"
#include "../../layout/descriptor.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor0::stride::cuda {

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA copy: ") + cudaGetErrorString(status));
  }
}

template <int Bytes>
void ExecuteCopy(const Storage<Bytes>* input, Storage<Bytes>* output,
    const descriptor::DecodedLayout& decoded,
    const std::vector<uint64_t>& counts, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
  std::size_t cursor = 4;
  for (std::size_t i = 0; i < decoded.records.size(); ++i) {
    const auto count = counts[i];
    const auto total = batches * count;
    if (total != 0) {
      const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
      CopyRecord<Bytes><<<blocks, 256, 0, stream>>>(input, output,
          device_descriptor + cursor, count, total, decoded.source_size, decoded.output_size);
      Check(cudaGetLastError());
    }
    cursor += 3 + 3 * decoded.records[i].shape.size();
  }
}

}  // namespace tensor0::stride::cuda
