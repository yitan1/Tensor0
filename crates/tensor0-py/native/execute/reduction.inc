template <typename Source, typename Accumulator, typename BindSource>
void ExecuteReduction(
    const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* accumulator,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    BindSource bind_source) {
  uint64_t source_elements = 0;
  uint64_t output_elements = 0;
  const auto maximum_bytes = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!layout::CheckedMultiply(source_size, batch_count, &source_elements) ||
      !layout::CheckedMultiply(output_size, batch_count, &output_elements) ||
      source_elements > maximum_bytes / sizeof(scalar::Value<Source>) ||
      output_elements > maximum_bytes / sizeof(scalar::Value<Accumulator>)) {
    throw std::invalid_argument("reduction batch storage exceeds address range");
  }
  if (output_elements == 0) return;
  std::fill_n(accumulator, output_elements, scalar::Value<Accumulator>{});
  for (const auto& record : records) {
    if (layout::ElementCount(record) == 0) continue;
    for (uint64_t batch = 0; batch < batch_count; ++batch) {
      bind_source(record.semantic_index, batch, [&](const auto& source_op) {
        using SourceOp = std::remove_cvref_t<decltype(source_op)>;
        static_assert(std::is_same_v<typename SourceOp::InputDtype, Source>);
        const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
        kernels::ExecuteReductionRecord<Accumulator>(
            record, source_batch, accumulator + batch * output_size, source_op);
      });
    }
  }
}
