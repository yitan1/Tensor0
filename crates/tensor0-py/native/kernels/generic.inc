namespace kernels {

template <std::size_t Count, typename RowOp>
void ForEachAffineRow(
    const layout::Record& record, std::array<int64_t, Count> offsets,
    const std::array<const std::vector<int64_t>*, Count>& strides, RowOp row_op) {
  const auto element_count = layout::ElementCount(record);
  if (element_count == 0) return;
  const uint64_t inner_count = record.shape.empty() ? 1 : record.shape[0];
  const uint64_t row_count = element_count / inner_count;
  std::vector<uint64_t> coordinates(record.shape.size());
  for (uint64_t row = 0; row < row_count; ++row) {
    row_op(offsets, inner_count);
    if (row + 1 == row_count) break;
    for (std::size_t axis = 1; axis < record.shape.size(); ++axis) {
      if (coordinates[axis] + 1 < record.shape[axis]) {
        ++coordinates[axis];
        for (std::size_t input = 0; input < Count; ++input) {
          offsets[input] += (*strides[input])[axis];
        }
        break;
      }
      for (std::size_t input = 0; input < Count; ++input) {
        offsets[input] -= static_cast<int64_t>(coordinates[axis]) * (*strides[input])[axis];
      }
      coordinates[axis] = 0;
    }
  }
}

template <typename SourceOp, typename Result = typename SourceOp::OutputDtype>
void ExecuteGenericAffineScalarRow(
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<Result>* result,
    int64_t source_base, int64_t destination_base, uint64_t inner_count,
    int64_t source_inner_stride, int64_t destination_inner_stride,
    const SourceOp& source_op) {
  int64_t source_index = source_base;
  int64_t destination_index = destination_base;
  for (uint64_t inner = 0; inner < inner_count; ++inner) {
    result[destination_index] =
        scalar::Convert<Result, typename SourceOp::OutputDtype>(source_op(source[source_index]));
    source_index += source_inner_stride;
    destination_index += destination_inner_stride;
  }
}

template <typename Dtype>
void ExecuteFillRecord(const layout::Record& record, scalar::Value<Dtype>* result,
                       scalar::Value<Dtype> value) {
  const int64_t inner_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<1>(record, {record.destination_offset}, {&record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t destination_index = offsets[0];
    if (inner_stride == 1) {
      std::fill_n(result + destination_index, inner_count, value);
    } else {
      for (uint64_t inner = 0; inner < inner_count; ++inner) {
        if (inner != 0) {
          destination_index += inner_stride;
        }
        result[destination_index] = value;
      }
    }
  });
}

template <typename Result, typename BinaryOp>
void ExecuteBinaryMapRecord(
    const layout::Record& record,
    const scalar::Value<typename BinaryOp::InputDtype>* source,
    const scalar::Value<typename BinaryOp::BaseInputDtype>* base,
    scalar::Value<Result>* result, const BinaryOp& operation) {
  const int64_t source_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t destination_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t source_index = offsets[0];
    int64_t destination_index = offsets[1];
    for (uint64_t inner = 0; inner < inner_count; ++inner) {
      const auto value = operation(source[source_index], base[destination_index]);
      result[destination_index] = scalar::Convert<Result, typename BinaryOp::OutputDtype>(value);
      source_index += source_stride;
      destination_index += destination_stride;
    }
  });
}

template <typename Accumulator, typename Mapped, typename ReadMapped>
void ExecuteReductionRow(
    scalar::Value<Accumulator>* accumulator, int64_t destination_index,
    int64_t destination_stride, uint64_t inner_count, ReadMapped read_mapped) {
  using Sum = scalar::Promote<Accumulator, Mapped>;
  static_assert(std::is_same_v<scalar::Promote<Sum, Mapped>, Sum>);
  if (destination_stride == 0) {
    auto sum = scalar::Add<Accumulator, Mapped>(accumulator[destination_index], read_mapped(0));
    for (uint64_t inner = 1; inner < inner_count; ++inner) {
      sum = scalar::Add<Sum, Mapped>(sum, read_mapped(inner));
    }
    accumulator[destination_index] = scalar::Convert<Accumulator, Sum>(sum);
    return;
  }
  for (uint64_t inner = 0; inner < inner_count; ++inner) {
    if (inner != 0) destination_index += destination_stride;
    const auto sum = scalar::Add<Accumulator, Mapped>(accumulator[destination_index], read_mapped(inner));
    accumulator[destination_index] = scalar::Convert<Accumulator, Sum>(sum);
  }
}

template <typename Accumulator, typename LeftOp, typename RightOp>
void ExecuteDotRecord(
    const layout::Record& record,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* accumulator,
    const LeftOp& left_op, const RightOp& right_op) {
  using Left = typename LeftOp::OutputDtype;
  using Right = typename RightOp::OutputDtype;
  using Product = scalar::Promote<Left, Right>;
  const int64_t left_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t right_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t left_index = offsets[0];
    int64_t right_index = offsets[1];
    ExecuteReductionRow<Accumulator, Product>(accumulator, 0, 0, inner_count, [&](uint64_t inner) {
      if (inner != 0) {
        left_index += left_stride;
        right_index += right_stride;
      }
      return scalar::Multiply<Left, Right>(left_op(left[left_index]), right_op(right[right_index]));
    });
  });
}

template <typename Accumulator, typename SourceOp>
void ExecuteReductionRecord(
    const layout::Record& record,
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<Accumulator>* accumulator, const SourceOp& source_op) {
  using Mapped = typename SourceOp::OutputDtype;
  const int64_t source_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t destination_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t source_index = offsets[0];
    ExecuteReductionRow<Accumulator, Mapped>(accumulator, offsets[1], destination_stride,
                                             inner_count, [&](uint64_t inner) {
      if (inner != 0) source_index += source_stride;
      return source_op(source[source_index]);
    });
  });
}

}
