namespace layout {

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

template <typename RecordType>
const auto& Strides(const RecordType& record, bool source) {
  return source ? record.source_strides : record.destination_strides;
}

template <typename RecordType>
int64_t Offset(const RecordType& record, bool source) {
  return source ? record.source_offset : record.destination_offset;
}

template <typename RecordType>
void AddressBounds(const RecordType& record, bool source,
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

template <typename RecordType>
void ValidateInjectiveView(const RecordType& record, bool source,
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


}
