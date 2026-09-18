template <typename Source, typename Result = Source>
void ExecuteCopyRecord(const layout::Record& record, const scalar::Value<Source>* source,
                       scalar::Value<Result>* result) {
  if constexpr (std::is_same_v<Source, Result>) {
    kernels::ExecuteMapRecord(record, source, result, expression::Identity<Source>{});
  } else {
    kernels::ExecuteMapRecord(
        record, source, result, expression::Cast<Result, expression::Identity<Source>>{});
  }
}

template <typename Source, typename Result = Source>
void ExecuteCopy(const std::vector<layout::Record>& records, const scalar::Value<Source>* source,
                 scalar::Value<Result>* result, uint64_t output_size) {
  if (output_size != 0) std::fill_n(result, output_size, scalar::Value<Result>{});
  for (const auto& record : records) {
    if (layout::ElementCount(record) == 0) continue;
    const auto program = layout::CompileGeneratedRecord(
        record, true, true, sizeof(scalar::Value<Source>), sizeof(scalar::Value<Result>));
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteCopyRecord<Source, Result>(block, source, result);
    });
  }
}
