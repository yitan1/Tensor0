// Preserve uncovered output storage; an in-place batch needs no initialization.
template <typename Result>
void InitializeUpdateOutput(const scalar::Value<Result>* base,
                            scalar::Value<Result>* result, uint64_t output_size) {
  if (output_size != 0 && result != base) {
    std::memcpy(result, base, output_size * sizeof(scalar::Value<Result>));
  }
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
void ExecuteUpdateRecord(
    const layout::Record& record,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<typename BaseMap::InputDtype>* base,
    scalar::Value<Result>* result,
    scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  if (layout::ElementCount(record) == 0) return;
  using ScaledSource = expression::Scale<Alpha, SourceMap>;
  using ScaledBase = expression::Scale<Beta, BaseMap>;
  if (scalar::IsZero<Alpha>(alpha)) {
    if (scalar::IsZero<Beta>(beta)) {
      kernels::ExecuteFillRecord<Result>(record, result, {});
    } else {
      auto base_record = record;
      base_record.source_offset = record.destination_offset;
      base_record.source_strides = record.destination_strides;
      if (scalar::IsOne<Beta>(beta)) {
        kernels::ExecuteMapRecord<BaseMap, Result>(base_record, base, result, base_map);
      } else {
        kernels::ExecuteMapRecord<ScaledBase, Result>(
            base_record, base, result, ScaledBase{beta, base_map});
      }
    }
    return;
  }
  if (scalar::IsZero<Beta>(beta)) {
    if (scalar::IsOne<Alpha>(alpha)) {
      kernels::ExecuteMapRecord<SourceMap, Result>(record, source, result, source_map);
    } else {
      kernels::ExecuteMapRecord<ScaledSource, Result>(
          record, source, result, ScaledSource{alpha, source_map});
    }
    return;
  }
  if (scalar::IsOne<Alpha>(alpha)) {
    if (scalar::IsOne<Beta>(beta)) {
      kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
          expression::AddUpdate<SourceMap, BaseMap>{source_map, base_map});
    } else {
      kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
          expression::AddUpdate<SourceMap, ScaledBase>{source_map, {beta, base_map}});
    }
  } else if (scalar::IsOne<Beta>(beta)) {
    kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
        expression::AddUpdate<ScaledSource, BaseMap>{{alpha, source_map}, base_map});
  } else {
    kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
        expression::AddUpdate<ScaledSource, ScaledBase>{{alpha, source_map}, {beta, base_map}});
  }
}

// Synchronously execute one complete batch, including output initialization.
template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
void ExecuteUpdate(
    const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t output_size, scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  static_assert(std::is_same_v<typename BaseMap::InputDtype, Result>);
  InitializeUpdateOutput<Result>(base, result, output_size);
  for (const auto& record : records) {
    if (layout::ElementCount(record) == 0) continue;
    const auto program = layout::CompileGeneratedRecord(
        record, true, true, sizeof(scalar::Value<typename SourceMap::InputDtype>),
        sizeof(scalar::Value<Result>));
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteUpdateRecord<Result, Alpha, Beta>(
          block, source, base, result, alpha, beta, source_map, base_map);
    });
  }
}

// Synchronously execute complete batches, initializing each output exactly once.
template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
void ExecuteUpdateBatches(
    const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    const scalar::Value<Alpha>* alpha, uint64_t alpha_count,
    const scalar::Value<Beta>* beta, uint64_t beta_count,
    const SourceMap& source_map, const BaseMap& base_map) {
  if ((alpha_count != 1 && alpha_count != batch_count) ||
      (beta_count != 1 && beta_count != batch_count)) {
    throw std::invalid_argument("update coefficients must be shared or one value per batch");
  }
  if (output_size == 0) return;
  for (uint64_t batch = 0; batch < batch_count; ++batch) {
    const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
    const auto* base_batch = base == nullptr ? nullptr : base + batch * output_size;
    ExecuteUpdate<Result, Alpha, Beta>(
        records, source_batch, base_batch, result + batch * output_size, output_size,
        alpha[alpha_count == 1 ? 0 : batch], beta[beta_count == 1 ? 0 : batch],
        source_map, base_map);
  }
}
