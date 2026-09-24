#pragma once

#include "../kernels/reduction.cuh"
#include "../../layout/descriptor.h"
#include "coefficient.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor0::stride::cuda::reduction {

struct Schedule {
  uint64_t owners, contributions;
  std::size_t cursor;
};

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA reduction: ") + cudaGetErrorString(status));
}

template <typename T, typename Real>
void Execute(const T* input, T* output, uint64_t source_size, uint64_t output_size,
    const std::vector<Schedule>& schedules,
    const std::vector<BoundCoefficient>& buffers,
    const std::vector<std::size_t>& parameters, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  // Bounds are validated for original axes. Optimization is never used to
  // interpret original device metadata; record indices stay semantic indices.
  // All validation, including empty-record coefficients, precedes writes.
  if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
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
        AccumulateRecord<T, Real, Raw><<<blocks, 256, 0, stream>>>(input, output, coefficient, count,
            reinterpret_cast<const uint64_t*>(device_descriptor) + schedule.cursor, schedule.owners, schedule.contributions, total,
            source_size, output_size);
        Check(cudaGetLastError());
      };
      if (parameter == buffers.size()) launch(std::type_identity<Real>{});
      else VisitCoefficient<T, Real>(buffers[parameter].dtype, launch);
    }
  }
}

}  // namespace tensor0::stride::cuda::reduction
