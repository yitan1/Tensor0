#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tensor0::stride {

inline uint64_t BufferBytes(uint64_t elements, uint64_t item_size) {
  if (elements > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / item_size) {
    throw std::invalid_argument("buffer size exceeds addressable storage");
  }
  return elements * item_size;
}

inline void ValidateDisjointBuffers(const void* input, uint64_t input_bytes,
                              const void* output, uint64_t output_bytes) {
  if (input_bytes == 0 || output_bytes == 0) return;
  const auto input_start = reinterpret_cast<std::uintptr_t>(input);
  const auto output_start = reinterpret_cast<std::uintptr_t>(output);
  const bool overlap = input_start <= output_start
      ? output_start - input_start < input_bytes
      : input_start - output_start < output_bytes;
  if (overlap) throw std::invalid_argument("input and output buffers overlap");
}

}  // namespace tensor0::stride
