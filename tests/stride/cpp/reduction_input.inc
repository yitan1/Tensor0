struct ReductionInput {
  std::size_t semantic_index = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  std::vector<uint64_t> shape;
  std::vector<int64_t> source_strides;
  std::vector<uint64_t> output_shape;
  std::vector<int64_t> destination_strides;
  std::vector<bool> reduction_axes;
};

layout::Record Build(const ReductionInput& input, uint64_t source_size,
                     uint64_t output_size) {
  return layout::BuildReductionLayout(
      input.shape, input.source_strides, input.source_offset,
      input.output_shape, input.destination_strides, input.destination_offset,
      input.reduction_axes, source_size, output_size, input.semantic_index);
}

layout::Record OutputView(const ReductionInput& input, uint64_t output_size) {
  const auto record = layout::BuildLayout(
      input.output_shape, input.destination_strides, input.destination_offset,
      input.destination_strides, input.destination_offset,
      output_size, output_size, input.semantic_index);
  std::vector<std::size_t> axes(layout::ElementCount(record) == 0 ? 0 : record.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  layout::ValidateInjectiveView(record, false, axes);
  return record;
}
