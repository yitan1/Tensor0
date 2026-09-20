#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"
#include "layout/traversal.h"
#include "layout/blocking.h"
#include "execute/scheduling.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "kernels/generic.h"
#include "kernels/specialized.h"
#include "kernels/dispatch.h"
#include "kernels/avx2.h"


using namespace tensor0::stride;
#include <cassert>
#include <cmath>
#include <cstring>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;











template <typename SourceOp>
void Check(const layout::Record& record, const std::vector<float>& source,
           uint64_t output_size, const SourceOp& operation) {
  std::vector<float> result(output_size, 7), expected = result;
  for (uint64_t element = 0; element < layout::ElementCount(record); ++element) {
    auto coordinates = element;
    auto source_index = record.source_offset;
    auto destination_index = record.destination_offset;
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      const auto coordinate = static_cast<int64_t>(coordinates % record.shape[axis]);
      coordinates /= record.shape[axis];
      source_index += coordinate * record.source_strides[axis];
      destination_index += coordinate * record.destination_strides[axis];
    }
    expected[destination_index] = operation(source[source_index]);
  }
  const auto check = [&] {
    for (std::size_t element = 0; element < result.size(); ++element) {
      if constexpr (std::is_same_v<SourceOp, expression::Identity<scalar::F32>>) {
        assert(std::bit_cast<uint32_t>(result[element]) == std::bit_cast<uint32_t>(expected[element]));
      } else if (std::isnan(expected[element])) {
        assert(std::isnan(result[element]));
      } else {
        assert(result[element] == expected[element]);
      }
    }
    std::fill(result.begin(), result.end(), 7);
  };
  kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
  check();
  auto optimized = record;
  layout::OptimizeRecordForExecution(&optimized);
  kernels::ExecuteMapRecord(optimized, source.data(), result.data(), operation);
  check();
  if (layout::ElementCount(record) == 0) return;
  std::array<std::size_t, 4> axes;
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (tensor0::stride::simd::CpuSupportsAvx2() && layout::IsRank4TwoPairCandidate(optimized, &axes)) {
    kernels::ExecuteRank4TwoPair(optimized, axes, source.data(), result.data(), operation);
    check();
  }
#endif
  const auto program = layout::CompileGeneratedRecord(optimized, true, true, sizeof(float), sizeof(float));
  layout::ForEachGeneratedBlock(program, [&](const auto& block) {
    kernels::ExecuteMapRecord(block, source.data(), result.data(), operation);
  });
  check();
}

int main() {
  const std::vector<uint32_t> bits{0, 0x80000000U, 0x3f800001U, 0xbf800001U,
      1, 0x00800000U, 0x7f7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00035U, 0x7f800035U};
  for (const auto dims : std::vector<std::array<uint64_t, 2>>{{8, 8}, {8, 16}, {24, 32}, {64, 64}, {7, 16}, {8, 9}, {0, 8}}) {
    for (uint64_t padding : {0, 5}) {
      const auto inner = dims[0] * dims[1];
      const auto pitch = inner + padding;
      std::vector<float> source(6 * inner + 4);
      for (std::size_t element = 0; element < source.size(); ++element) source[element] = std::bit_cast<float>(bits[element % bits.size()]);
      const auto output_size = 6 * pitch + 4;
      auto record = layout::BuildLayout({2, 3, dims[0], dims[1]},
          {static_cast<int64_t>(inner), static_cast<int64_t>(2 * inner), 1, static_cast<int64_t>(dims[0])}, 1,
          {static_cast<int64_t>(3 * pitch), static_cast<int64_t>(pitch), static_cast<int64_t>(dims[1]), 1}, 2,
          source.size(), output_size, 0);
      std::array<std::size_t, 4> axes;
      const bool eligible = dims[0] != 0 && dims[0] % 8 == 0 && dims[1] % 8 == 0;
      assert(layout::IsRank4TwoPairCandidate(record, &axes) == eligible);
      auto optimized = record;
      layout::OptimizeRecordForExecution(&optimized);
      assert(layout::IsRank4TwoPairCandidate(optimized, &axes) == eligible);
      Check(record, source, output_size, expression::Identity<scalar::F32>{});
      for (float factor : {0.F, 1.F, -1.25F, std::numeric_limits<float>::infinity()}) {
        Check(record, source, output_size, expression::Scale<scalar::F32>{factor, {}});
      }
      if (eligible && inner <= 128) {
        std::array<std::size_t, 4> permutation{0, 1, 2, 3};
        do {
          auto reordered = record;
          for (std::size_t axis = 0; axis < 4; ++axis) {
            reordered.shape[axis] = record.shape[permutation[axis]];
            reordered.source_strides[axis] = record.source_strides[permutation[axis]];
            reordered.destination_strides[axis] = record.destination_strides[permutation[axis]];
          }
          assert(layout::IsRank4TwoPairCandidate(reordered, &axes));
          Check(reordered, source, output_size, expression::Identity<scalar::F32>{});
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        const auto program = layout::CompileGeneratedRecord(optimized, true, true, sizeof(float), sizeof(float));
        bool matched = false;
        layout::ForEachGeneratedBlock(program, [&](const auto& block) {
          matched |= layout::IsRank4TwoPairCandidate(block, &axes);
        });
        assert(matched);
      }
      record.source_offset += static_cast<int64_t>(dims[0] == 0 ? 0 : dims[0] - 1);
      record.source_strides[2] = -1;
      assert(!layout::IsRank4TwoPairCandidate(record, &axes));
      Check(record, source, output_size, expression::Identity<scalar::F32>{});
    }
  }
}
