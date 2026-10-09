#pragma once

#include "../layout/descriptor.h"
#include "../layout/blocking.h"
#include "buffers.h"
#include "xla/ffi/api/ffi.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace tensor0::stride {

namespace ffi = xla::ffi;

extern std::atomic<uint64_t> prepared_created_count;
extern std::atomic<uint64_t> prepared_destroyed_count;

struct PreparedState {
  static ffi::TypeId id;
  explicit PreparedState(descriptor::DecodedLayout value, bool outputs_disjoint = false);
  PreparedState(const PreparedState&) = delete;
  PreparedState& operator=(const PreparedState&) = delete;
  ~PreparedState();

  const std::vector<layout::Record> records;
  const uint64_t source_size;
  const uint64_t output_size;
  const bool outputs_disjoint;
  const uint64_t disjoint_work;

  std::shared_ptr<const std::vector<layout::GeneratedRecordProgram>> AccumulationPrograms(
      uint64_t source_item_size, uint64_t result_item_size) const;

 private:
  mutable std::mutex accumulation_mutex;
  mutable std::map<std::pair<uint64_t, uint64_t>,
                   std::shared_ptr<const std::vector<layout::GeneratedRecordProgram>>> accumulation_programs;
};

extern const ffi::TypeInfo kPreparedTypeInfo;

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(ffi::Span<const int64_t> words);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateReduction(
    ffi::Span<const uint8_t> bytes, ffi::Span<const int64_t>);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateDot(
    ffi::Span<const int64_t> words, int64_t);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateAccumulation(
    ffi::Span<const int64_t> words, ffi::Span<const int64_t>, int64_t outputs_disjoint = 0);

void ValidatePreparedDimensions(const PreparedState& prepared, uint64_t source_size, uint64_t output_size);

}
