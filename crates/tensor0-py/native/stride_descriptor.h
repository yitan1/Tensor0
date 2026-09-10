#ifndef TENSOR0_STRIDE_DESCRIPTOR_H_
#define TENSOR0_STRIDE_DESCRIPTOR_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace tensor0::stride {

static_assert(sizeof(uint64_t) == 8);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(float) == 4);

// Little-endian uint64 ABI:
//
//   header:
//     magic, version, total_words, source_size, required_source_size,
//     output_size, record_count, coverage, source_dtype, result_dtype,
//     scalar_policy, execution_flags
//
//   records:
//     rank, source_offset, destination_offset, scale_real_bits,
//     scale_imaginary_bits, identity_scale, mapped_dtype,
//     shape[rank], signed_source_strides[rank],
//     signed_destination_strides[rank]
//
// Signed strides use two's-complement int64 words. Semantic records are
// variable-rank; native instantiate removes singleton axes and fuses jointly
// contiguous axes before applying kMaximumRank to the prepared executor.
// A zero source stride on a nontrivial axis denotes broadcast directly.
// Destination zero strides remain invalid. Each record is checked independently
// for bounds, overflow, and destination injectivity.
// Cross-record uniqueness and complete coverage are semantic preconditions
// established by the layout/transform planner.
inline constexpr uint64_t kDescriptorMagic =
    UINT64_C(0x3150444952543054);
inline constexpr uint64_t kDescriptorVersion = 9;
inline constexpr uint64_t kHeaderWords = 12;
inline constexpr uint64_t kCoverageCompleteUnique = 1;
inline constexpr uint64_t kCoveragePartialUniqueZeroFill = 2;
inline constexpr uint64_t kDtypeF32 = 1;
inline constexpr uint64_t kDtypeF16 = 2;
inline constexpr uint64_t kDtypeBF16 = 3;
inline constexpr uint64_t kDtypeC64 = 4;
inline constexpr uint64_t kDtypeS32 = 5;
inline constexpr uint64_t kDtypeF64 = 6;
inline constexpr uint64_t kDtypeC128 = 7;
inline constexpr uint64_t kDtypePred = 8;
inline constexpr uint64_t kDtypeS8 = 9;
inline constexpr uint64_t kDtypeS16 = 10;
inline constexpr uint64_t kDtypeS64 = 11;
inline constexpr uint64_t kDtypeU8 = 12;
inline constexpr uint64_t kDtypeU16 = 13;
inline constexpr uint64_t kDtypeU32 = 14;
inline constexpr uint64_t kDtypeU64 = 15;
inline constexpr std::size_t kMaximumRank = 8;

inline constexpr uint64_t kScalarForwardScaleCast = 1;
inline constexpr uint64_t kScalarJaxTranspose = 2;
// Native structured-reduction semantic ABI-v4:
//
//   header:
//     magic, version, total_words, input_size, output_size, record_count,
//     input_dtype, output_dtype, scalar_policy
//
//   records:
//     rank, input_offset, output_offset, scale_real_bits,
//     scale_imaginary_bits, identity_scale, mapped_dtype,
//     shape[rank], signed_input_strides[rank],
//     signed_output_strides[rank], reduction_axis_flags[rank]
//
// CPU normalization, loop orders, fiber counts, grouping, and chunk policy
// are compiled exclusively by native instantiate.
inline constexpr uint64_t kReductionDescriptorMagic =
    UINT64_C(0x3152444952543054);
inline constexpr uint64_t kReductionDescriptorVersion = 4;
inline constexpr uint64_t kReductionDescriptorHeaderWords = 9;
inline constexpr uint64_t kReductionScalarForwardScaleCast =
    kScalarForwardScaleCast;
inline constexpr uint64_t kReductionScalarJaxTranspose =
    kScalarJaxTranspose;

// Native two-input affine dot semantic ABI-v1:
//
//   header:
//     magic, version, total_words, left_size, right_size, record_count,
//     dtype, conjugate_left
//
//   records:
//     rank, left_offset, right_offset, logical_elements, shape[rank],
//     signed_left_strides[rank], signed_right_strides[rank]
inline constexpr uint64_t kDotDescriptorMagic =
    UINT64_C(0x31544f4449523054);
inline constexpr uint64_t kDotDescriptorVersion = 1;
inline constexpr uint64_t kDotDescriptorHeaderWords = 8;

struct Record {
  std::size_t semantic_index = 0;
  std::size_t rank = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
  float scale = 1.0F;
  bool identity_scale = true;
  uint64_t mapped_dtype = 0;
  uint64_t logical_elements = 0;
  std::array<uint64_t, kMaximumRank> shape{};
  std::array<int64_t, kMaximumRank> source_strides{};
  std::array<int64_t, kMaximumRank> destination_strides{};
  std::array<std::size_t, kMaximumRank> loop_axes_fastest_first{};
};

}  // namespace tensor0::stride

#endif  // TENSOR0_STRIDE_DESCRIPTOR_H_
