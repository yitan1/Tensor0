#pragma once

#include "descriptor.h"

#include <algorithm>

namespace tensor0::stride::layout {

struct OwnerFiberSchedule {
  uint64_t owners;
  uint64_t contributions;
  std::size_t cursor;
};

struct OwnerFiberLayout {
  std::vector<int64_t> words;
  std::vector<OwnerFiberSchedule> schedules;
};

constexpr uint64_t kDotChunk = 1024;
inline uint64_t DotChunks(uint64_t contributions) {
  return contributions <= 1 ? 0 : (contributions - 1) / kDotChunk + 1;
}
// The sequential strategy remains valid for huge unsigned reduction fibers;
// allocating scratch for every possible contribution is not required.
constexpr uint64_t kParallelScratchLimit = 1 << 20;
inline uint64_t SumChunks(const OwnerFiberSchedule& schedule) {
  if (schedule.owners == 0 || schedule.contributions < kDotChunk) return 0;
  const uint64_t chunks = (schedule.contributions - 1) / kDotChunk + 1;
  return chunks <= kParallelScratchLimit / schedule.owners ? chunks : 0;
}
inline uint64_t SumScratchCapacity(const std::vector<OwnerFiberSchedule>& schedules) {
  uint64_t capacity = 0;
  for (const auto& schedule : schedules)
    capacity = std::max(capacity, schedule.owners * SumChunks(schedule));
  return capacity;
}
inline uint64_t DotScratchCapacity(const std::vector<OwnerFiberSchedule>& schedules) {
  uint64_t capacity = 0;
  for (const auto& schedule : schedules)
    capacity = std::max(capacity, DotChunks(schedule.contributions));
  return capacity;
}

// Mechanical device encoding of fully validated semantic layouts. The input
// record order, offsets, and each group's relative axis order are preserved.
OwnerFiberLayout PackOwnerFiber(const descriptor::DecodedLayout& decoded, int32_t operation = 3);
OwnerFiberLayout PackOwnerFiber(const descriptor::DecodedReductionLayout& decoded);

}  // namespace tensor0::stride::layout

// Internal CUDA descriptor bridge. Input is a validated/prepared semantic
// descriptor; complete decoding is repeated for raw-FFI safety, never sorting.
extern "C" void Tensor0StridePackOwnerFiber(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept;

extern "C" void Tensor0StrideSumScratchCapacity(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept;

extern "C" void Tensor0StrideDotScratchCapacity(
    const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept;
