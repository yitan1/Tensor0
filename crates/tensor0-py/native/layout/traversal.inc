namespace layout {

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

void OptimizeRecordForExecution(Record* record) {
  if (ElementCount(*record) == 0 || record->shape.size() == 0) {
    SortRecordDimensions(record);
    return;
  }

  SortRecordDimensions(record);
  for (std::size_t axis = record->shape.size() - 1; axis > 0; --axis) {
    int64_t source_span = 0;
    int64_t destination_span = 0;
    if (!CheckedMultiplyStride(record->source_strides[axis - 1],
                               record->shape[axis - 1], &source_span) ||
        !CheckedMultiplyStride(record->destination_strides[axis - 1],
                               record->shape[axis - 1], &destination_span) ||
        source_span != record->source_strides[axis] ||
        destination_span != record->destination_strides[axis]) {
      continue;
    }
    uint64_t merged_shape = 0;
    if (CheckedMultiply(record->shape[axis - 1], record->shape[axis], &merged_shape)) {
      record->shape[axis - 1] = merged_shape;
      record->shape[axis] = 1;
    }
  }
  SortRecordDimensions(record);
  while (!record->shape.empty() && record->shape.back() == 1) {
    record->shape.pop_back();
    record->source_strides.pop_back();
    record->destination_strides.pop_back();
  }
}

bool IsCompactSameMapping(const Record& record) {
  if (record.shape.size() == 0) {
    return true;
  }
  std::vector<std::size_t> order(record.shape.size());
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if (record.source_strides[axis] != record.destination_strides[axis] ||
        record.source_strides[axis] <= 0) {
      return false;
    }
    order[axis] = axis;
  }
  std::stable_sort(
      order.begin(), order.begin() + record.shape.size(),
      [&](std::size_t left, std::size_t right) {
        return record.source_strides[left] < record.source_strides[right];
      });
  uint64_t expected = 1;
  for (std::size_t position = 0; position < record.shape.size(); ++position) {
    const std::size_t axis = order[position];
    if (record.shape[axis] <= 1) {
      continue;
    }
    if (record.source_strides[axis] != static_cast<int64_t>(expected) ||
        !CheckedMultiply(expected, record.shape[axis], &expected)) {
      return false;
    }
  }
  return true;
}

bool IsRank2ForwardCandidate(const Record& record) {
  return record.shape.size() == 2 && record.source_strides[0] == 1 &&
         StrideEquals(record.source_strides[1], record.shape[0]) &&
         record.destination_strides[1] == 1 &&
         StrideEquals(record.destination_strides[0], record.shape[1]);
}

bool IsRank2ReverseCandidate(const Record& record) {
  return record.shape.size() == 2 && record.source_strides[1] == 1 &&
         StrideEquals(record.source_strides[0], record.shape[1]) &&
         record.destination_strides[0] == 1 &&
         StrideEquals(record.destination_strides[1], record.shape[0]);
}

uint64_t GenericRowCount(const Record& record) {
  return ElementCount(record) /
         record.shape[0];
}

bool IsBroadcastContiguousRowCandidate(const Record& record) {
  if (record.shape.size() == 0) {
    return false;
  }
  constexpr std::size_t fastest_axis = 0;
  return record.shape[fastest_axis] > 1 &&
         record.source_strides[fastest_axis] == 0 &&
         record.destination_strides[fastest_axis] == 1;
}

bool IsContiguousInnerRowsCandidate(const Record& record) {
  if (record.shape.size() == 0) {
    return false;
  }
  constexpr std::size_t fastest_axis = 0;
  return record.shape[fastest_axis] > 1 &&
         record.source_strides[fastest_axis] == 1 &&
         record.destination_strides[fastest_axis] == 1;
}

bool IsRank4TwoPairCandidate(const Record& record, std::array<std::size_t, 4>* axes) {
  if (record.shape.size() != 4) return false;
  std::array<std::size_t, 4> order{0, 1, 2, 3};
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return record.source_strides[left] < record.source_strides[right];
  });
  const auto axis0 = order[2];
  const auto axis1 = order[3];
  const auto axis2 = order[0];
  const auto axis3 = order[1];
  if (record.shape[axis2] == 0 || record.shape[axis3] == 0 ||
      record.shape[axis2] % 8 != 0 || record.shape[axis3] % 8 != 0) {
    return false;
  }
  uint64_t inner = 0;
  uint64_t source_axis1 = 0;
  uint64_t destination_axis0 = 0;
  uint64_t destination_axis1_stride = 0;
  const bool matches = CheckedMultiply(record.shape[axis2], record.shape[axis3], &inner) &&
         CheckedMultiply(record.shape[axis0], inner, &source_axis1) &&
         PositiveStrideValue(record.destination_strides[axis1],
                             &destination_axis1_stride) &&
         CheckedMultiply(record.shape[axis1], destination_axis1_stride,
                         &destination_axis0) &&
         StrideEquals(record.source_strides[axis0], inner) &&
         StrideEquals(record.source_strides[axis1], source_axis1) &&
         record.source_strides[axis2] == 1 &&
         StrideEquals(record.source_strides[axis3], record.shape[axis2]) &&
         record.destination_strides[axis3] == 1 &&
         StrideEquals(record.destination_strides[axis2], record.shape[axis3]) &&
         destination_axis1_stride >= inner &&
         StrideEquals(record.destination_strides[axis0], destination_axis0);
  if (matches) *axes = {axis0, axis1, axis2, axis3};
  return matches;
}

}
