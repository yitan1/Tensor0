#pragma once

#include "../../layout/descriptor.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tensor0::stride::cuda {

// A single validated injective record covers every output slot exactly once
// when its sorted destination strides form a dense affine span. Multiple
// records are deliberately not inferred to be disjoint here.
inline bool FullyCoversOutput(const descriptor::DecodedLayout& decoded) {
  if (decoded.records.size() != 1) return false;
  const auto& record = decoded.records.front();
  if (layout::ElementCount(record) != decoded.output_size) return false;
  if (decoded.output_size == 0) return true;
  std::vector<std::size_t> axes(record.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::sort(axes.begin(), axes.end(), [&](auto a, auto b) {
    return layout::AbsoluteStride(record.destination_strides[a]) <
           layout::AbsoluteStride(record.destination_strides[b]);
  });
  uint64_t span = 1;
  int64_t minimum = record.destination_offset;
  for (auto axis : axes) {
    if (record.shape[axis] <= 1) continue;
    if (layout::AbsoluteStride(record.destination_strides[axis]) != span) return false;
    if (record.destination_strides[axis] < 0)
      minimum += static_cast<int64_t>(record.shape[axis] - 1) * record.destination_strides[axis];
    span *= record.shape[axis];
  }
  return minimum == 0 && span == decoded.output_size;
}

}  // namespace tensor0::stride::cuda
