#pragma once

#include "generic.h"
#include "specialized.h"
#include "../layout/traversal.h"
#include "../numeric/expression.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace tensor0::stride::kernels {

template <typename SourceOp>
inline constexpr bool kIsIdentityMapping = false;

template <typename Dtype>
inline constexpr bool kIsIdentityMapping<expression::Identity<Dtype>> = true;

template <typename SourceOp, typename Result = typename SourceOp::OutputDtype>
void ExecuteMapRecord(
    const layout::Record& record,
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<Result>* result,
    const SourceOp& source_op) {
  const auto element_count = layout::ElementCount(record);
  if (element_count == 0) {
    return;
  }
  if constexpr (std::is_same_v<Result, typename SourceOp::OutputDtype>) {
    if (layout::IsCompactSameMapping(record)) {
      const auto* input = source + record.source_offset;
      auto* output = result + record.destination_offset;
      if constexpr (kIsIdentityMapping<SourceOp>) {
        if (input != output) {
          std::memcpy(output, input, element_count * sizeof(*source));
        }
      } else {
        ExecuteContiguousMap(input, output, element_count, source_op);
      }
      return;
    }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
    if constexpr (std::is_same_v<SourceOp, expression::Identity<scalar::F32>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::F32>>) {
      std::array<std::size_t, 4> axes;
      if (::tensor0::stride::simd::CpuSupportsAvx2() &&
          layout::IsRank4TwoPairCandidate(record, &axes)) {
        ExecuteRank4TwoPair(record, axes, source, result, source_op);
        return;
      }
    }
    if constexpr (std::is_same_v<SourceOp, expression::Identity<scalar::F16>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::F16>> ||
                  std::is_same_v<SourceOp, expression::Identity<scalar::C64>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
      constexpr bool is_half = std::is_same_v<typename SourceOp::InputDtype, scalar::F16>;
      constexpr uint64_t vector_size = is_half ? 8 : 4;
      if (record.shape.size() == 2 && record.shape[0] >= vector_size && record.shape[1] >= vector_size &&
          (is_half ? ::tensor0::stride::simd::CpuSupportsAvx2F16c()
                   : ::tensor0::stride::simd::CpuSupportsAvx2Fma())) {
        for (std::size_t axis = 0; axis < 2; ++axis) {
          const auto other = 1 - axis;
          if (record.source_strides[axis] == 1 && record.destination_strides[other] == 1 &&
              record.source_strides[other] > 0 && record.destination_strides[axis] > 0 &&
              static_cast<uint64_t>(record.source_strides[other]) >= record.shape[axis] &&
              static_cast<uint64_t>(record.destination_strides[axis]) >= record.shape[other]) {
            if constexpr (is_half) {
              ExecuteRank2F16Transpose(source + record.source_offset, result + record.destination_offset,
                  record.shape[axis], record.shape[other], record.source_strides[other],
                  record.destination_strides[axis], source_op);
            } else {
              ExecuteRank2C64Transpose(source + record.source_offset, result + record.destination_offset,
                  record.shape[axis], record.shape[other], record.source_strides[other],
                  record.destination_strides[axis], source_op);
            }
            return;
          }
        }
      }
    }
#endif
    if (layout::IsRank2ForwardCandidate(record)) {
      ExecuteRank2AffineColumns(record, source, result, source_op);
      return;
    }
    if (layout::IsRank2ReverseCandidate(record)) {
      ExecuteRank2AffineRows(record, source, result, source_op);
      return;
    }
  }
  const int64_t source_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t destination_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    const int64_t source_base = offsets[0];
    const int64_t destination_base = offsets[1];
    if (inner_count > 1 && source_stride == 0 && destination_stride == 1) {
      const auto mapped = scalar::Convert<Result, typename SourceOp::OutputDtype>(
          source_op(source[source_base]));
      std::fill_n(result + destination_base, inner_count, mapped);
    } else if constexpr (std::is_same_v<Result, typename SourceOp::OutputDtype>) {
      if (source_stride == 1 && destination_stride == 1) {
        ExecuteContiguousMap(source + source_base, result + destination_base, inner_count, source_op);
      } else {
        ExecuteGenericAffineScalarRow<SourceOp, Result>(
            source, result, source_base, destination_base, inner_count,
            source_stride, destination_stride, source_op);
      }
    } else {
      if (source_stride == 1 && destination_stride == 1) {
        ExecuteContiguousMap(source + source_base, result + destination_base, inner_count,
            expression::Cast<Result, SourceOp>{source_op});
      } else {
        ExecuteGenericAffineScalarRow<SourceOp, Result>(
            source, result, source_base, destination_base, inner_count,
            source_stride, destination_stride, source_op);
      }
    }
  });
}

}
