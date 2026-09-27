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
// Record order represents semantic_index and must remain unchanged.
struct DecodedLayout {
  std::vector<layout::Record> records;
  uint64_t source_size = 0;
  uint64_t output_size = 0;
};

struct DecodedReductionLayout {
  std::vector<layout::ReductionRecord> records;
  uint64_t source_size = 0;
  uint64_t output_size = 0;
};

// Address validation permits repeated source and destination addresses.
DecodedLayout DecodeAddressLayout(const int64_t* words, std::size_t word_count);

// Accumulation permits zero-stride fibers but requires provably disjoint output owners.
DecodedLayout DecodeAccumulationLayout(const int64_t* words, std::size_t word_count);

// Map semantics additionally require injective destinations within each record.
DecodedLayout DecodeLayout(const int64_t* words, std::size_t word_count);

// Validate the unsigned reduction protocol, including axis roles and empty fibers.
DecodedReductionLayout DecodeReductionLayout(const char* bytes, std::size_t byte_count);

// Shared, device-independent preparation validates the complete descriptor before
// stable locality sorting, first-axis-fastest fusion and re-sorting. Reduction
// retains axis roles and only fuses matching roles; empty records stay intact.
DecodedLayout PrepareLayout(const int64_t* words, std::size_t word_count);
DecodedLayout PrepareAddressLayout(const int64_t* words, std::size_t word_count);
DecodedLayout PrepareAccumulationLayout(const int64_t* words, std::size_t word_count);
DecodedReductionLayout PrepareReductionLayout(const char* bytes, std::size_t byte_count);

// Internal V1 semantic encoders require validated, protocol-representable
// layouts, including shared axis optimization and unchanged record order. Arbitrary mutations are not supported; encoders do not revalidate.
// Reduction output shapes are reconstructed from roles; ignored destination
// strides are emitted in canonical form, not necessarily as originally received.
std::vector<int64_t> EncodeLayout(const DecodedLayout& decoded);
std::vector<char> EncodeReductionLayout(const DecodedReductionLayout& decoded);

}

}  // namespace tensor0::stride

// Synchronous, allocation-owning caller bridge. Operations: 0=copy, 1=update,
// 2=dot, 3=accumulation, 4=reduction. All words are signed i64; reduction words
// carry unsigned protocol bit patterns and are serialized little-endian.
// The same shared preparation functions used by CPU instantiate validate the
// entire descriptor before common sort/fuse/sort normalization and encoding.
// Empty records remain unchanged; ignored reduction strides are canonicalized
// during decoding.
// The callback is called exactly once: status 0=success, 1=invalid descriptor or
// operation, 2=internal failure. Buffers are borrowed only during the callback;
// it must copy synchronously, must not throw/unwind, and must not retain pointers.
// callback must be non-null; words may be null only when word_count is zero.
using Tensor0StridePrepareLayoutCallback = void (*)(
    void* context, int32_t status, const int64_t* words, std::size_t word_count,
    const char* error, std::size_t error_size);
extern "C" void Tensor0StridePrepareLayout(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept;
