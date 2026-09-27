#include "record.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensor0::stride::layout {

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t* result) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

uint64_t ElementCount(const Record& record) {
  uint64_t count = 1;
  for (const auto extent : record.shape) {
    if (!CheckedMultiply(count, extent, &count)) {
      throw std::invalid_argument("layout element count overflows");
    }
  }
  return count;
}

bool HasIdenticalAddresses(const Record& record) {
  if (record.source_offset != record.destination_offset) return false;
  const bool empty = ElementCount(record) == 0;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if ((empty || record.shape[axis] > 1) &&
        record.source_strides[axis] != record.destination_strides[axis]) {
      return false;
    }
  }
  return true;
}

uint64_t AbsoluteStride(int64_t stride) {
  if (stride == std::numeric_limits<int64_t>::min()) {
    return UINT64_C(1) << 63;
  }
  return stride < 0 ? static_cast<uint64_t>(-stride)
                    : static_cast<uint64_t>(stride);
}

bool StrideEquals(int64_t stride, uint64_t value) {
  return value <=
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         stride == static_cast<int64_t>(value);
}

bool PositiveStrideValue(int64_t stride, uint64_t* value) {
  if (stride <= 0) {
    return false;
  }
  *value = static_cast<uint64_t>(stride);
  return true;
}

bool CheckedMultiplyStride(int64_t stride, uint64_t extent,
                           int64_t* result) {
  uint64_t magnitude = 0;
  if (!CheckedMultiply(AbsoluteStride(stride), extent, &magnitude) ||
      magnitude >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t signed_magnitude = static_cast<int64_t>(magnitude);
  *result = stride < 0 ? -signed_magnitude : signed_magnitude;
  return true;
}

namespace {

bool TryFuseAdjacentPair(Record& record, std::size_t outer,
                         uint64_t maximum_extent) {
  const std::size_t inner = outer + 1;
  if (record.shape[outer] <= 1 || record.shape[inner] <= 1) {
    return false;
  }
  uint64_t extent = 0;
  int64_t source_stride = 0;
  int64_t destination_stride = 0;
  if (!CheckedMultiply(record.shape[outer], record.shape[inner], &extent) ||
      extent > maximum_extent ||
      !CheckedMultiplyStride(record.source_strides[inner], record.shape[inner],
                             &source_stride) ||
      source_stride != record.source_strides[outer] ||
      !CheckedMultiplyStride(record.destination_strides[inner], record.shape[inner],
                             &destination_stride) ||
      destination_stride != record.destination_strides[outer]) {
    return false;
  }
  record.shape[outer] = extent;
  record.source_strides[outer] = record.source_strides[inner];
  record.destination_strides[outer] = record.destination_strides[inner];
  record.shape.erase(record.shape.begin() + inner);
  record.source_strides.erase(record.source_strides.begin() + inner);
  record.destination_strides.erase(record.destination_strides.begin() + inner);
  return true;
}

}  // namespace

void FuseAdjacentAxes(Record& record) {
  if (ElementCount(record) == 0) return;
  for (std::size_t axis = 0; axis + 1 < record.shape.size();) {
    if (!TryFuseAdjacentPair(record, axis,
                             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))) {
      ++axis;
    }
  }
}

void FuseAdjacentAxes(ReductionRecord& record) {
  if (ElementCount(record.layout) == 0) return;
  for (std::size_t axis = 0; axis + 1 < record.layout.shape.size();) {
    if (record.reduction_axes[axis] == record.reduction_axes[axis + 1] &&
        TryFuseAdjacentPair(record.layout, axis, std::numeric_limits<uint64_t>::max())) {
      record.reduction_axes.erase(record.reduction_axes.begin() + axis + 1);
    } else {
      ++axis;
    }
  }
}

std::vector<uint64_t> GeneratedIndexOrder(
    const std::vector<int64_t>& strides) {
  std::vector<uint64_t> order(strides.size(), 1);
  for (std::size_t axis = 0; axis < strides.size(); ++axis) {
    const uint64_t stride = AbsoluteStride(strides[axis]);
    if (stride == 0) continue;
    for (int64_t candidate : strides) {
      const uint64_t candidate_stride = AbsoluteStride(candidate);
      if (candidate_stride != 0 && candidate_stride < stride) ++order[axis];
    }
  }
  return order;
}

std::vector<std::size_t> ComputeLocalityOrder(const Record& record) {
  const auto destination_order = GeneratedIndexOrder(record.destination_strides);
  const auto source_order = GeneratedIndexOrder(record.source_strides);
  std::vector<std::size_t> axes(record.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  const auto importance_key = [&](std::size_t axis) {
    const auto destination = destination_order[axis];
    const auto source = source_order[axis];
    const uint64_t priority = destination == source ? 0 : destination < source ? 1 : 2;
    return std::array<uint64_t, 4>{
        record.shape[axis] > 1 ? UINT64_C(0) : UINT64_C(1),
        record.shape[axis] > 1 ? std::min(destination, source) : 0,
        record.shape[axis] > 1 ? priority : 0,
        record.shape[axis] > 1 ? std::max(destination, source) : 0};
  };
  std::stable_sort(axes.begin(), axes.end(), [&](std::size_t left, std::size_t right) {
    return importance_key(left) < importance_key(right);
  });
  return axes;
}

void SortRecordDimensions(Record* record) {
  const auto axes = ComputeLocalityOrder(*record);
  const auto reorder = [&](auto* values) {
    auto reordered = *values;
    for (std::size_t axis = 0; axis < record->shape.size(); ++axis) {
      reordered[axis] = (*values)[axes[axis]];
    }
    *values = std::move(reordered);
  };
  reorder(&record->shape);
  reorder(&record->source_strides);
  reorder(&record->destination_strides);
}

namespace {

template <typename T>
void ReorderAxes(std::vector<T>* values, const std::vector<std::size_t>& order) {
  auto reordered = *values;
  for (std::size_t axis = 0; axis < order.size(); ++axis) {
    reordered[axis] = (*values)[order[axis]];
  }
  *values = std::move(reordered);
}

void SortPreparedAxes(Record& layout, std::vector<bool>* roles) {
  const auto axes = ComputeLocalityOrder(layout);
  ReorderAxes(&layout.shape, axes);
  ReorderAxes(&layout.source_strides, axes);
  ReorderAxes(&layout.destination_strides, axes);
  if (roles) ReorderAxes(roles, axes);
}

void OptimizePreparedAxes(Record& layout, std::vector<bool>* roles,
                          uint64_t maximum_extent) {
  // Empty layouts retain every axis, including ignored singleton strides.
  if (ElementCount(layout) == 0) return;
  SortPreparedAxes(layout, roles);
  for (std::size_t axis = layout.shape.size(); axis > 1; --axis) {
    const std::size_t inner = axis - 2;
    const std::size_t outer = axis - 1;
    int64_t source_span = 0;
    int64_t destination_span = 0;
    uint64_t extent = 0;
    if (roles && (*roles)[inner] != (*roles)[outer]) continue;
    if (!CheckedMultiplyStride(layout.source_strides[inner], layout.shape[inner],
                               &source_span) ||
        !CheckedMultiplyStride(layout.destination_strides[inner], layout.shape[inner],
                               &destination_span) ||
        source_span != layout.source_strides[outer] ||
        destination_span != layout.destination_strides[outer] ||
        !CheckedMultiply(layout.shape[inner], layout.shape[outer], &extent) ||
        extent > maximum_extent) continue;
    layout.shape[inner] = extent;
    layout.shape[outer] = 1;
  }
  SortPreparedAxes(layout, roles);
  while (!layout.shape.empty() && layout.shape.back() == 1) {
    layout.shape.pop_back();
    layout.source_strides.pop_back();
    layout.destination_strides.pop_back();
    if (roles) roles->pop_back();
  }
}

}  // namespace

void OptimizeRecordForPreparation(Record& record) {
  if (ElementCount(record) == 0) return;
  // Removing singleton/fused axes changes locality ranks of the remaining
  // strides. Repeat until rank stops decreasing so re-preparation is stable.
  for (;;) {
    const auto rank = record.shape.size();
    OptimizePreparedAxes(record, nullptr, static_cast<uint64_t>(INT64_MAX));
    if (record.shape.size() == rank) return;
  }
}

void OptimizeRecordForPreparation(ReductionRecord& record) {
  if (ElementCount(record.layout) == 0) return;
  for (;;) {
    const auto rank = record.layout.shape.size();
    OptimizePreparedAxes(record.layout, &record.reduction_axes, UINT64_MAX);
    if (record.layout.shape.size() == rank) return;
  }
}

const std::vector<int64_t>& Strides(const Record& record, bool source) {
  return source ? record.source_strides : record.destination_strides;
}

int64_t Offset(const Record& record, bool source) {
  return source ? record.source_offset : record.destination_offset;
}

void AddressBounds(const Record& record, bool source,
                   const std::vector<std::size_t>& axes,
                   int64_t* minimum, int64_t* maximum) {
  if (Offset(record, source) < 0) {
    throw std::invalid_argument(std::string(source ? "source" : "destination") +
                                " address precedes storage");
  }
  int64_t low = Offset(record, source);
  int64_t high = low;
  const auto& strides = Strides(record, source);
  for (const auto axis : axes) {
    if (record.shape[axis] <= 1) continue;
    uint64_t magnitude = 0;
    if (!CheckedMultiply(record.shape[axis] - 1,
                         AbsoluteStride(strides[axis]), &magnitude) ||
        magnitude >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::invalid_argument(std::string(source ? "source" : "destination") +
                     " address arithmetic overflow");
    }
    if (strides[axis] < 0) {
      if (magnitude > static_cast<uint64_t>(low)) {
        throw std::invalid_argument(std::string(source ? "source" : "destination") +
                       " address precedes storage");
      }
      low -= static_cast<int64_t>(magnitude);
    } else {
      if (magnitude > static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max() - high)) {
        throw std::invalid_argument(std::string(source ? "source" : "destination") +
                       " address arithmetic overflow");
      }
      high += static_cast<int64_t>(magnitude);
    }
  }
  *minimum = low;
  *maximum = high;
}

void ValidateInjectiveView(const Record& record, bool source,
                           std::vector<std::size_t> axes) {
  const auto& strides = Strides(record, source);
  std::stable_sort(
      axes.begin(), axes.end(),
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(strides[left]), left) <
               std::make_pair(AbsoluteStride(strides[right]), right);
      });

  uint64_t covered_span = 0;
  for (const auto axis : axes) {
    const uint64_t extent = record.shape[axis];
    const uint64_t stride = AbsoluteStride(strides[axis]);
    if (extent <= 1) {
      continue;
    }
    if (stride <= covered_span) {
      throw std::invalid_argument(std::string(source ? "source" : "destination") +
                     " strides do not prove an injective view");
    }
    uint64_t axis_span = 0;
    if (!CheckedMultiply(extent - 1, stride, &axis_span) ||
        !CheckedAdd(covered_span, axis_span, &covered_span)) {
      throw std::invalid_argument("injectivity arithmetic overflow");
    }
  }
}

Record BuildLayout(
    const std::vector<uint64_t>& shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<int64_t>& destination_strides, int64_t destination_offset,
    uint64_t source_size, uint64_t output_size, std::size_t semantic_index) {
  if (source_strides.size() != shape.size() || destination_strides.size() != shape.size()) {
    throw std::invalid_argument("layout field lengths mismatch");
  }
  Record record{semantic_index, source_offset, destination_offset, shape,
                source_strides, destination_strides};
  const auto element_count = ElementCount(record);
  std::vector<std::size_t> axes(element_count == 0 ? 0 : record.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  for (bool source : {true, false}) {
    int64_t minimum = 0;
    int64_t maximum = 0;
    AddressBounds(record, source, axes, &minimum, &maximum);
    const uint64_t size = source ? source_size : output_size;
    if ((element_count == 0 && static_cast<uint64_t>(maximum) > size) ||
        (element_count != 0 && static_cast<uint64_t>(maximum) >= size)) {
      throw std::invalid_argument(std::string(source ? "source" : "destination") +
                                  " address exceeds storage");
    }
  }
  return record;
}

Record BuildReductionLayout(
    const std::vector<uint64_t>& source_shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<uint64_t>& output_shape,
    const std::vector<int64_t>& output_strides, int64_t output_offset,
    const std::vector<bool>& reduction_axes, uint64_t source_size,
    uint64_t output_size, std::size_t semantic_index) {
  if (source_strides.size() != source_shape.size() ||
      output_shape.size() != source_shape.size() ||
      output_strides.size() != source_shape.size() ||
      reduction_axes.size() != source_shape.size()) {
    throw std::invalid_argument("reduction layout field lengths mismatch");
  }
  Record record{semantic_index, source_offset, output_offset, source_shape,
                source_strides, output_strides};
  if (source_offset < 0 || output_offset < 0) {
    throw std::invalid_argument("reduction address precedes storage");
  }
  uint64_t output_count = 1;
  uint64_t reduction_count = 1;
  std::vector<std::size_t> map_axes;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    const uint64_t extent = record.shape[axis];
    const bool reduction = reduction_axes[axis];
    if (output_shape[axis] != (reduction ? 1 : extent)) {
      throw std::invalid_argument("reduction input/output shapes do not match axis roles");
    }
    if (reduction) record.destination_strides[axis] = 0;
    if (!reduction && extent > 1 && record.destination_strides[axis] == 0) {
      throw std::invalid_argument("reduction output map has a zero nontrivial stride");
    }
    uint64_t* count = reduction ? &reduction_count : &output_count;
    if (!CheckedMultiply(*count, extent, count)) {
      throw std::invalid_argument("reduction fiber element count overflow");
    }
    if (!reduction) map_axes.push_back(axis);
  }
  uint64_t logical_count = 0;
  if (!CheckedMultiply(output_count, reduction_count, &logical_count)) {
    throw std::invalid_argument("reduction logical element count overflow");
  }
  ValidateInjectiveView(record, false, map_axes);
  std::vector<std::size_t> source_axes(record.shape.size());
  std::iota(source_axes.begin(), source_axes.end(), 0);
  for (bool source : {true, false}) {
    const bool empty = source ? logical_count == 0 : output_count == 0;
    const uint64_t size = source ? source_size : output_size;
    if (empty) {
      if (static_cast<uint64_t>(Offset(record, source)) > size) {
        throw std::invalid_argument("empty reduction offset exceeds storage");
      }
    } else {
      int64_t minimum = 0;
      int64_t maximum = 0;
      AddressBounds(record, source, source ? source_axes : map_axes,
                    &minimum, &maximum);
      if (static_cast<uint64_t>(maximum) >= size) {
        throw std::invalid_argument("reduction address exceeds storage");
      }
    }
  }
  return record;
}

Record SliceRecordRange(const Record& record, std::size_t axis,
                            uint64_t unit_start, uint64_t unit_count) {
  if (axis >= record.shape.size() || unit_start > record.shape[axis] ||
      unit_count > record.shape[axis] - unit_start) {
    throw std::invalid_argument("record range exceeds layout extent");
  }
  Record slice = record;
  slice.shape[axis] = unit_count;
  if (ElementCount(record) == 0 || unit_count == 0) {
    return slice;
  }
  slice.source_offset +=
      static_cast<int64_t>(unit_start) * record.source_strides[axis];
  slice.destination_offset +=
      static_cast<int64_t>(unit_start) * record.destination_strides[axis];
  return slice;
}

}
