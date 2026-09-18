namespace kernels {

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
