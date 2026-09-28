#pragma once

#include "../../layout/owner_fiber.h"

#include <cstdint>

namespace tensor0::stride::cuda::owner_fiber {

// Packed V1 has two address lanes. Dot uses the second lane for its right
// input; the other operations use it for their output address.
__device__ __forceinline__ void Address(const int64_t* record, uint64_t logical,
                                         bool fiber, int64_t& first, int64_t& second) {
  const auto map_rank = static_cast<uint64_t>(record[0]);
  const auto rank = map_rank + static_cast<uint64_t>(record[1]);
  const auto begin = fiber ? map_rank : 0;
  const auto end = fiber ? rank : map_rank;
  // The caller's logical coordinate is already within the sole axis extent.
  // Avoid a 64-bit remainder and division for one-axis owner/fiber groups.
  if (end - begin == 1) {
    const auto first_stride = record[4 + rank + begin];
    const auto second_stride = record[4 + 2 * rank + begin];
    if (first_stride != 0) first += static_cast<int64_t>(logical) * first_stride;
    if (second_stride != 0) second += static_cast<int64_t>(logical) * second_stride;
    return;
  }
  for (uint64_t axis = end; axis-- > begin;) {
    const auto coordinate = logical % static_cast<uint64_t>(record[4 + axis]);
    logical /= static_cast<uint64_t>(record[4 + axis]);
    const auto first_stride = record[4 + rank + axis];
    const auto second_stride = record[4 + 2 * rank + axis];
    // A reduction's unsigned fiber extent can exceed the signed address range
    // only when its corresponding input stride is zero.
    if (first_stride != 0) first += static_cast<int64_t>(coordinate) * first_stride;
    if (second_stride != 0) second += static_cast<int64_t>(coordinate) * second_stride;
  }
}

// Operation-specific expressions are selected statically. Only this kernel
// owns the owner/fiber coordinates, traversal order and launch grid walk.
template <typename Policy>
__global__ void MapReduceRecord(Policy policy, const int64_t* record,
                                uint64_t owners, uint64_t contributions, uint64_t total) {
  const uint64_t step = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;) {
    const uint64_t batch = index / owners;
    const auto context = policy.Begin(batch);
    if (policy.Active(context)) {
      int64_t first = record[2], second = record[3];
      Address(record, index % owners, false, first, second);
      auto value = policy.Initial(context, batch, second);
      for (uint64_t logical = 0; logical < contributions; ++logical) {
        int64_t input_first = first, input_second = second;
        Address(record, logical, true, input_first, input_second);
        value = policy.Combine(value, policy.Term(context, batch, input_first, input_second));
      }
      policy.Store(value, batch, second);
    }
    if (total - index <= step) break;
    index += step;
  }
}

// Parallel fiber execution shares the same packed address lanes and static
// contribution policy as MapReduceRecord. The output is combined only once,
// after all chunks belonging to a record have completed on the same stream.
template <typename Policy, bool SingleOwner = false, bool Direct = false>
__global__ void PartialFibers(Policy policy, const int64_t* record,
                              typename Policy::Value* scratch, uint64_t contributions,
                              uint64_t owners, uint64_t chunks, uint64_t capacity, uint64_t tasks) {
  using Value = typename Policy::Value;
  __shared__ Value shared[256];
  for (uint64_t task = blockIdx.x; task < tasks;) {
    const uint64_t batch = SingleOwner ? task / chunks : task / (owners * chunks);
    const uint64_t owner = SingleOwner ? 0 : task / chunks % owners;
    const uint64_t chunk = task % chunks;
    const auto context = policy.Begin(batch);
    if (SingleOwner || policy.Active(context)) {
      const uint64_t begin = chunk * layout::kDotChunk;
      const uint64_t length = min(layout::kDotChunk, contributions - begin);
      int64_t base_first = record[2], base_second = record[3];
      if constexpr (!SingleOwner) Address(record, owner, false, base_first, base_second);
      Value value{};
      for (uint64_t local = threadIdx.x; local < length; local += blockDim.x) {
        int64_t first = base_first, second = base_second;
        Address(record, begin + local, true, first, second);
        value = policy.Combine(value, policy.Term(context, batch, first, second));
      }
      shared[threadIdx.x] = value;
      __syncthreads();
      for (unsigned width = 128; width != 0; width /= 2) {
        if (threadIdx.x < width)
          shared[threadIdx.x] = policy.Combine(shared[threadIdx.x], shared[threadIdx.x + width]);
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        if constexpr (Direct) {
          policy.Store(policy.Combine(policy.Initial(context, batch, base_second), shared[0]), batch, base_second);
        } else {
          scratch[batch * capacity + owner * chunks + chunk] = shared[0];
        }
      }
      __syncthreads();
    }
    if (tasks - task <= gridDim.x) break;
    task += gridDim.x;
  }
}

template <typename Policy, bool SingleOwner = false>
__global__ void FinishFibers(Policy policy, const typename Policy::Value* scratch,
                             const int64_t* record, uint64_t owners, uint64_t chunks,
                             uint64_t capacity, uint64_t tasks) {
  using Value = typename Policy::Value;
  __shared__ Value shared[256];
  for (uint64_t task = blockIdx.x; task < tasks;) {
    const uint64_t batch = SingleOwner ? task : task / owners;
    const uint64_t owner = SingleOwner ? 0 : task % owners;
    const auto context = policy.Begin(batch);
    if (SingleOwner || policy.Active(context)) {
      Value value{};
      for (uint64_t i = threadIdx.x; i < chunks;) {
        value = policy.Combine(value, scratch[batch * capacity + owner * chunks + i]);
        if (chunks - i <= blockDim.x) break;
        i += blockDim.x;
      }
      shared[threadIdx.x] = value;
      __syncthreads();
      for (unsigned width = 128; width != 0; width /= 2) {
        if (threadIdx.x < width)
          shared[threadIdx.x] = policy.Combine(shared[threadIdx.x], shared[threadIdx.x + width]);
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        int64_t first = record[2], second = record[3];
        if constexpr (!SingleOwner) Address(record, owner, false, first, second);
        policy.Store(policy.Combine(policy.Initial(context, batch, second), shared[0]), batch, second);
      }
      __syncthreads();
    }
    if (tasks - task <= gridDim.x) break;
    task += gridDim.x;
  }
}

}  // namespace tensor0::stride::cuda::owner_fiber
