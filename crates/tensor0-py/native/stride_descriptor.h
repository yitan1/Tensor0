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
//     output_size, record_count, coverage, dtype
//
//   records:
//     rank, source_offset, destination_offset, scale_real_bits,
//     scale_imaginary_bits, source_broadcast_axis_mask,
//     shape[rank], signed_source_strides[rank],
//     signed_destination_strides[rank]
//
// Signed strides use two's-complement int64 words. A zero source stride on a
// nontrivial axis is legal only when the corresponding explicit broadcast bit
// is set. Destination zero strides remain invalid. Each record is checked
// independently for bounds, overflow, and destination injectivity.
// Cross-record uniqueness and complete coverage are semantic preconditions
// established by the layout/transform planner.
inline constexpr uint64_t kDescriptorMagic =
    UINT64_C(0x3150444952543054);
inline constexpr uint64_t kDescriptorVersion = 5;
inline constexpr uint64_t kHeaderWords = 9;
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

// Current compiled descriptor. It embeds one complete version-5 scalar-domain
// witness plus the true source/result dtypes and scalar policy, followed by
// Python's normalized execution decisions. Native instantiate revalidates both
// views before publishing executable-owned state.
inline constexpr uint64_t kCompiledDescriptorMagic =
    UINT64_C(0x3150434952543054);
inline constexpr uint64_t kCompiledDescriptorVersion = 7;
inline constexpr uint64_t kCompiledDescriptorHeaderWords = 14;
inline constexpr uint64_t kCompilerPolicyVersion = 3;
inline constexpr uint64_t kCpuPolicyVersion = 7;
inline constexpr uint64_t kRawDescriptorWitness = 1;
inline constexpr uint64_t kScalarForwardScaleCast = 1;
inline constexpr uint64_t kScalarJaxTranspose = 2;

// Native structured-reduction ABI-v2. Records carry an explicit map/reduction
// axis partition and fixed loop orders; the scalar-policy field distinguishes
// forward scale/cast from the JAX linear-transpose scalar rule. Native
// instantiate independently revalidates the affine fibers before publishing
// prepared state.
inline constexpr uint64_t kReductionDescriptorMagic =
    UINT64_C(0x3152444952543054);
inline constexpr uint64_t kReductionDescriptorVersion = 2;
inline constexpr uint64_t kReductionCompilerPolicyVersion = 3;
inline constexpr uint64_t kReductionDescriptorHeaderWords = 13;
inline constexpr uint64_t kReductionScalarForwardScaleCast =
    kScalarForwardScaleCast;
inline constexpr uint64_t kReductionScalarJaxTranspose =
    kScalarJaxTranspose;

struct Record {
  std::size_t rank = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
  uint64_t source_broadcast_axis_mask = 0;
  float scale = 1.0F;
  uint64_t logical_elements = 0;
  std::array<uint64_t, kMaximumRank> shape{};
  std::array<int64_t, kMaximumRank> source_strides{};
  std::array<int64_t, kMaximumRank> destination_strides{};
  std::array<std::size_t, kMaximumRank> loop_axes_fastest_first{};
};

}  // namespace tensor0::stride

#endif  // TENSOR0_STRIDE_DESCRIPTOR_H_
