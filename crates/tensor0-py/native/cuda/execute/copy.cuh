#pragma once

#include "../kernels/owner_fiber.cuh"
#include "../kernels/map_reduce_policies.cuh"
#include "../../layout/owner_fiber.h"
#include "../../layout/descriptor.h"
#include "check.cuh"
#include "full_coverage.cuh"

#include <algorithm>
#include <vector>

namespace tensor0::stride::cuda {

template <int Bytes>
void ExecuteCopy(const Storage<Bytes>* input, Storage<Bytes>* output,
    const descriptor::DecodedLayout& decoded,
    const std::vector<layout::OwnerFiberSchedule>& schedules, uint64_t batches,
    uint64_t output_bytes, bool full_coverage,
    const int64_t* device_descriptor, cudaStream_t stream) {
  if (output_bytes != 0 && !full_coverage)
    CheckCuda(cudaMemsetAsync(output, 0, output_bytes, stream), "copy");
  for (const auto schedule : schedules) {
    const auto total = batches * schedule.owners;
    if (total != 0) {
      const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
      owner_fiber::MapReduceRecord<<<blocks, 256, 0, stream>>>(
          CopyMap<Bytes>{input, output, decoded.source_size, decoded.output_size},
          device_descriptor + schedule.cursor, schedule.owners, schedule.contributions, total);
      CheckCuda(cudaGetLastError(), "copy");
    }
  }
}

}  // namespace tensor0::stride::cuda
