#pragma once

#include "../kernels/owner_fiber.cuh"
#include "../kernels/map_reduce_policies.cuh"
#include "../../layout/owner_fiber.h"
#include "../../layout/descriptor.h"
#include "check.cuh"

#include <algorithm>
#include <vector>

namespace tensor0::stride::cuda::dot {

template <typename T>
void Execute(const T* left, const T* right, T* output, T* scratch, bool conjugate,
    const descriptor::DecodedLayout& decoded,
    const std::vector<layout::OwnerFiberSchedule>& schedules, uint64_t capacity,
    uint64_t batches, uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  if (output_bytes != 0) CheckCuda(cudaMemsetAsync(output, 0, output_bytes, stream), "dot");
  for (const auto schedule : schedules) {
    if (batches != 0 && schedule.contributions != 0) {
      const auto policy = DotFiber<T>{left, right, output, decoded.source_size, decoded.output_size, conjugate};
      const auto chunks = layout::DotChunks(schedule.contributions);
      if (chunks == 0) {
        const auto blocks = static_cast<unsigned>(std::min<uint64_t>((batches - 1) / 256 + 1, 65535));
        owner_fiber::MapReduceRecord<<<blocks, 256, 0, stream>>>(
            policy, device_descriptor + schedule.cursor, schedule.owners, schedule.contributions, batches);
        CheckCuda(cudaGetLastError(), "dot");
      } else {
        const uint64_t tasks = batches * chunks;
        owner_fiber::PartialFibers<DotFiber<T>, true><<<static_cast<unsigned>(std::min<uint64_t>(tasks, 65535)), 256, 0, stream>>>(
            policy, device_descriptor + schedule.cursor, scratch, schedule.contributions,
            schedule.owners, chunks, capacity, tasks);
        CheckCuda(cudaGetLastError(), "dot");
        owner_fiber::FinishFibers<DotFiber<T>, true><<<static_cast<unsigned>(std::min<uint64_t>(batches, 65535)), 256, 0, stream>>>(
            policy, scratch, device_descriptor + schedule.cursor,
            schedule.owners, chunks, capacity, batches);
        CheckCuda(cudaGetLastError(), "dot");
      }
    }
  }
}

}  // namespace tensor0::stride::cuda::dot
