#pragma once

#include "../layout/descriptor.h"
#include "buffers.h"
#include "xla/ffi/api/ffi.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tensor0::stride {

namespace ffi = xla::ffi;

extern std::atomic<uint64_t> prepared_created_count;
extern std::atomic<uint64_t> prepared_destroyed_count;

struct PreparedState {
  static ffi::TypeId id;
  explicit PreparedState(descriptor::DecodedLayout value);
  PreparedState(const PreparedState&) = delete;
  PreparedState& operator=(const PreparedState&) = delete;
  ~PreparedState();

  const std::vector<layout::Record> records;
  const uint64_t source_size;
  const uint64_t output_size;
};

extern const ffi::TypeInfo kPreparedTypeInfo;

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(ffi::Span<const int64_t> words);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateReduction(
    ffi::Span<const uint8_t> bytes, ffi::Span<const int64_t>);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateDot(
    ffi::Span<const int64_t> words, int64_t);

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateAccumulation(
    ffi::Span<const int64_t> words, ffi::Span<const int64_t>);

void ValidatePreparedDimensions(const PreparedState& prepared, uint64_t source_size, uint64_t output_size);

}
