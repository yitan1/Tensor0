#pragma once

#include "../kernels/update.cuh"
#include "../../layout/descriptor.h"
#include "coefficient.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor0::stride::cuda::update {

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA update: ") + cudaGetErrorString(status));
  }
}

template <typename T, typename Real>
void Execute(const T* input, const T* old, T* output,
    BoundCoefficient alpha, BoundCoefficient beta,
    const descriptor::DecodedLayout& decoded,
    const std::vector<uint64_t>& counts, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  // All metadata and overlap validation precedes any device writes.
  VisitCoefficient<T, Real>(alpha.dtype, [&](auto alpha_type) {
    using Alpha = typename decltype(alpha_type)::type;
    VisitCoefficient<T, Real>(beta.dtype, [&](auto beta_type) {
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
              input, old, output, reinterpret_cast<const Alpha*>(alpha.data), alpha.count,
              reinterpret_cast<const Beta*>(beta.data), beta.count,
              device_descriptor + cursor, counts[i], total, decoded.source_size, decoded.output_size);
          Check(cudaGetLastError());
        }
        cursor += 3 + 3 * decoded.records[i].shape.size();
      }
    });
  });
}

}  // namespace tensor0::stride::cuda::update
