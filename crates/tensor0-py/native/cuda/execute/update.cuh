#pragma once

#include "../kernels/owner_fiber.cuh"
#include "../kernels/map_reduce_policies.cuh"
#include "../../layout/owner_fiber.h"
#include "../../layout/descriptor.h"
#include "check.cuh"
#include "full_coverage.cuh"
#include "coefficient.cuh"

#include <algorithm>
#include <vector>

namespace tensor0::stride::cuda::update {

template <typename T, typename Real>
void Execute(const T* input, const T* old, T* output,
    BoundCoefficient alpha, BoundCoefficient beta,
    const descriptor::DecodedLayout& decoded,
    const std::vector<layout::OwnerFiberSchedule>& schedules, uint64_t batches,
    uint64_t output_bytes, bool full_coverage,
    const int64_t* device_descriptor, cudaStream_t stream) {
  // All metadata and overlap validation precedes any device writes.
  arithmetic::VisitCoefficient<T, Real>(alpha.dtype, [&](auto alpha_type) {
    using Alpha = typename decltype(alpha_type)::type;
    arithmetic::VisitCoefficient<T, Real>(beta.dtype, [&](auto beta_type) {
      using Beta = typename decltype(beta_type)::type;
      if (output_bytes != 0 && old != output && !full_coverage) {
        CheckCuda(cudaMemcpyAsync(output, old, output_bytes, cudaMemcpyDeviceToDevice, stream), "update");
      }
      for (const auto schedule : schedules) {
        const auto total = batches * schedule.owners;
        if (total != 0) {
          const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
          owner_fiber::MapReduceRecord<<<blocks, 256, 0, stream>>>(
              UpdateMap<T, Real, Alpha, Beta>{input, old, output,
                  reinterpret_cast<const Alpha*>(alpha.data), alpha.count,
                  reinterpret_cast<const Beta*>(beta.data), beta.count,
                  decoded.source_size, decoded.output_size},
              device_descriptor + schedule.cursor, schedule.owners, schedule.contributions, total);
          CheckCuda(cudaGetLastError(), "update");
        }
      }
    });
  });
}

}  // namespace tensor0::stride::cuda::update
