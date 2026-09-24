#pragma once

#include "../kernels/accumulation.cuh"
#include "../../layout/descriptor.h"
#include "coefficient.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor0::stride::cuda::accumulation {

struct Schedule {
  uint64_t owners, contributions;
  bool general;
};

Schedule Classify(const layout::Record& record, uint64_t count) {
  if (count == 0) return {0, 0, false};
  std::vector<std::size_t> map_axes;
  uint64_t owners = 1;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if (record.shape[axis] > 1 && record.destination_strides[axis] != 0) {
      map_axes.push_back(axis);
      // ElementCount has already checked the product of all nonempty extents.
      owners *= record.shape[axis];
    }
  }
  try {
    layout::ValidateInjectiveView(record, false, std::move(map_axes));
    return {owners, count / owners, false};
  } catch (const std::invalid_argument&) {
    // Failure of the sufficient proof is not proof of a collision.
    return {1, count, true};
  }
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA accumulation: ") + cudaGetErrorString(status));
}

template <typename T, typename Real>
void Execute(const T* input, T* output, const descriptor::DecodedLayout& decoded,
    const std::vector<Schedule>& schedules,
    const std::vector<BoundCoefficient>& buffers,
    const std::vector<std::size_t>& parameters, uint64_t batches,
    uint64_t output_bytes, const int64_t* device_descriptor, cudaStream_t stream) {
  // DecodeAddressLayout bounds each positive/negative span independently from
  // the offset. Every partial coordinate sum, in any axis order, is within
  // those signed bounds; original-axis device address arithmetic is safe.
  // All validation, including empty-record coefficients, precedes writes.
  if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
  std::size_t cursor = 4;
  for (std::size_t i = 0; i < decoded.records.size(); ++i) {
    const auto schedule = schedules[i];
    const auto total = batches * schedule.owners;
    if (total != 0) {
      const auto blocks = static_cast<unsigned>(std::min<uint64_t>((total - 1) / 256 + 1, 65535));
      const auto parameter = parameters[i];
      auto launch = [&](auto type) {
        using Raw = typename decltype(type)::type;
        const Raw* coefficient = parameter == buffers.size() ? nullptr : reinterpret_cast<const Raw*>(buffers[parameter].data);
        const uint64_t count = parameter == buffers.size() ? 1 : buffers[parameter].count;
        AccumulateRecord<T, Real, Raw><<<blocks, 256, 0, stream>>>(input, output, coefficient, count,
            device_descriptor + cursor, schedule.owners, schedule.contributions, total,
            schedule.general, decoded.source_size, decoded.output_size);
        Check(cudaGetLastError());
      };
      if (parameter == buffers.size()) launch(std::type_identity<Real>{});
      else VisitCoefficient<T, Real>(buffers[parameter].dtype, launch);
    }
    cursor += 3 + 3 * decoded.records[i].shape.size();
  }
}

}  // namespace tensor0::stride::cuda::accumulation
