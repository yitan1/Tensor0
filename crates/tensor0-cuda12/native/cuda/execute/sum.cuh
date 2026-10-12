#pragma once

#include "../kernels/owner_fiber.cuh"
#include "../kernels/map_reduce_policies.cuh"
#include "../../layout/owner_fiber.h"
#include "check.cuh"
#include "coefficient.cuh"

#include <algorithm>
#include <vector>

namespace tensor0::stride::cuda::sum {

template <typename T, typename Real>
void Execute(const T* input, T* output, T* scratch, uint64_t capacity,
    uint64_t source_size, uint64_t output_size,
    const std::vector<layout::OwnerFiberSchedule>& schedules,
    const std::vector<BoundCoefficient>& buffers,
    const std::vector<std::size_t>& parameters, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream,
    const char* operation) {
  // All validation, including empty-record coefficients, precedes writes.
  if (output_bytes != 0) CheckCuda(cudaMemsetAsync(output, 0, output_bytes, stream), operation);
  for (std::size_t i = 0; i < schedules.size(); ++i) {
    const auto schedule = schedules[i];
    const auto total = batches * schedule.owners;
    if (total != 0 && schedule.contributions != 0) {
      const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
      const auto parameter = parameters[i];
      auto launch = [&](auto type) {
        using Raw = typename decltype(type)::type;
        const Raw* coefficient = parameter == buffers.size() ? nullptr : reinterpret_cast<const Raw*>(buffers[parameter].data);
        const uint64_t count = parameter == buffers.size() ? 1 : buffers[parameter].count;
        const auto policy = SumFiber<T, Real, Raw>{input, output, coefficient, count, source_size, output_size};
        const auto chunks = layout::SumChunks(schedule);
        if (chunks == 0) {
          owner_fiber::MapReduceRecord<<<blocks, 256, 0, stream>>>(
              policy, device_descriptor + schedule.cursor, schedule.owners, schedule.contributions, total);
          CheckCuda(cudaGetLastError(), operation);
        } else {
          const auto tasks = total * chunks;
          owner_fiber::PartialFibers<<<static_cast<unsigned>(std::min<uint64_t>(tasks, 65535)), 256, 0, stream>>>(
              policy, device_descriptor + schedule.cursor, scratch, schedule.contributions,
              schedule.owners, chunks, capacity, tasks);
          CheckCuda(cudaGetLastError(), operation);
          owner_fiber::FinishFibers<<<static_cast<unsigned>(std::min<uint64_t>(total, 65535)), 256, 0, stream>>>(
              policy, scratch, device_descriptor + schedule.cursor,
              schedule.owners, chunks, capacity, total);
          CheckCuda(cudaGetLastError(), operation);
        }
      };
      if (parameter == buffers.size()) launch(std::type_identity<Real>{});
      else arithmetic::VisitCoefficient<T, Real>(buffers[parameter].dtype, launch);
    }
  }
}

}  // namespace tensor0::stride::cuda::sum
