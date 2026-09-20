#pragma once

#include "../layout/record.h"
#include "xla/ffi/api/ffi.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tensor0::stride {

namespace ffi = xla::ffi;

namespace descriptor {

inline constexpr int64_t kLayoutVersion = 1;
inline constexpr uint64_t kReductionLayoutVersion = 1;

struct DecodedLayout {
  std::vector<layout::Record> records;
  uint64_t source_size = 0;
  uint64_t output_size = 0;
};

DecodedLayout DecodeAddressLayout(const int64_t* words, std::size_t word_count);

DecodedLayout DecodeLayout(const int64_t* words, std::size_t word_count);

DecodedLayout DecodeReductionLayout(const char* bytes, std::size_t byte_count);

}

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

uint64_t BufferBytes(uint64_t elements, uint64_t item_size);

void ValidateDisjointBuffers(const void* input, uint64_t input_bytes,
                              const void* output, uint64_t output_bytes);

void ValidatePreparedDimensions(const PreparedState& prepared, uint64_t source_size, uint64_t output_size);

}
