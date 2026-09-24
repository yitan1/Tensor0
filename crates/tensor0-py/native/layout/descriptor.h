#pragma once

#include "record.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor0::stride {

namespace descriptor {

inline constexpr int64_t kLayoutVersion = 1;
inline constexpr uint64_t kReductionLayoutVersion = 1;

// Validated semantic records retain axis order and singleton axes. They are not
// CPU-optimized execution plans or a lossless copy of the wire protocol:
// reduction records canonicalize reduction-axis destination strides to zero.
struct DecodedLayout {
  std::vector<layout::Record> records;
  uint64_t source_size = 0;
  uint64_t output_size = 0;
};

// Address validation permits repeated source and destination addresses.
DecodedLayout DecodeAddressLayout(const int64_t* words, std::size_t word_count);

// Map semantics additionally require injective destinations within each record.
DecodedLayout DecodeLayout(const int64_t* words, std::size_t word_count);

// Validate the unsigned reduction protocol, including axis roles and empty fibers.
DecodedLayout DecodeReductionLayout(const char* bytes, std::size_t byte_count);

}

}  // namespace tensor0::stride
