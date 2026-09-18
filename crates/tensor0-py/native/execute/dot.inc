template <typename Accumulator, typename LeftOp, typename RightOp>
void ExecuteDot(
    const std::vector<layout::Record>& records,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* result,
    uint64_t left_size, uint64_t right_size, uint64_t batch_count,
    const LeftOp& left_op, const RightOp& right_op) {
  uint64_t left_elements = 0;
  uint64_t right_elements = 0;
  const auto maximum_bytes = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!layout::CheckedMultiply(left_size, batch_count, &left_elements) ||
      !layout::CheckedMultiply(right_size, batch_count, &right_elements) ||
      left_elements > maximum_bytes / sizeof(scalar::Value<typename LeftOp::InputDtype>) ||
      right_elements > maximum_bytes / sizeof(scalar::Value<typename RightOp::InputDtype>) ||
      batch_count > maximum_bytes / sizeof(scalar::Value<Accumulator>)) {
    throw std::invalid_argument("dot batch storage exceeds address range");
  }
  if (batch_count == 0) return;
  std::fill_n(result, batch_count, scalar::Value<Accumulator>{});
  for (const auto& record : records) {
    if (layout::ElementCount(record) == 0) continue;
    for (uint64_t batch = 0; batch < batch_count; ++batch) {
      kernels::ExecuteDotRecord<Accumulator>(
          record, left + batch * left_size, right + batch * right_size,
          result + batch, left_op, right_op);
    }
  }
}
