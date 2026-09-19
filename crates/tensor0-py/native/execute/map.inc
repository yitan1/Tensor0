ffi::Future ExecuteMapTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::GeneratedRecordProgram>& prepared,
    std::function<void()> initialize,
    std::function<void(const layout::Record&)> execute) {
  try {
    uint64_t elements = 0;
    for (const auto& program : prepared) {
      if (!layout::CheckedAdd(elements, layout::ElementCount(program.record), &elements)) {
        throw std::invalid_argument("map work size overflows");
      }
    }
    const auto workers = std::min(
        AvailableWorkerCount(thread_pool),
        std::max<uint64_t>(1, elements / (UINT64_C(1) << 15)));
    std::vector<layout::GeneratedRecordProgram> programs;
    for (const auto& program : prepared) {
      layout::AppendGeneratedSubdomains(program, workers, &programs);
    }
    // Initialize the complete batch once, before any subdomain can write.
    initialize();
    const auto count = programs.size();
    return ExecuteTasks(thread_pool, count, workers,
        [programs = std::move(programs), execute = std::move(execute)](uint64_t begin, uint64_t count) {
          for (uint64_t index = begin; index < begin + count; ++index) {
            layout::ForEachGeneratedBlock(programs[index], execute);
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map task exception"));
  }
}

// Initialize the entire batch, never an individual record or parallel subdomain.
template <typename Result>
void InitializeCopyOutput(scalar::Value<Result>* result, uint64_t output_size) {
  if (output_size != 0) std::fill_n(result, output_size, scalar::Value<Result>{});
}

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

// Synchronously execute one complete batch, including output initialization.
template <typename Source, typename Result = Source>
void ExecuteCopyBatch(const std::vector<layout::GeneratedRecordProgram>& programs, const scalar::Value<Source>* source,
                 scalar::Value<Result>* result, uint64_t output_size) {
  InitializeCopyOutput<Result>(result, output_size);
  for (const auto& program : programs) {
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteCopyRecord<Source, Result>(block, source, result);
    });
  }
}

template <typename Source, typename Result>
ffi::Future ExecuteCopy(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count) {
  try {
    auto programs = layout::PrepareGeneratedRecords(records, sizeof(*source), sizeof(*result));
    if (batch_count == 1 && output_size != 0) {
      return ExecuteMapTasks(thread_pool, programs,
          [=] { InitializeCopyOutput<Result>(result, output_size); },
          [=](const auto& record) { ExecuteCopyRecord<Source, Result>(record, source, result); });
    }
    return ExecuteBatchTasks(thread_pool, batch_count, output_size,
        [=, programs = std::move(programs)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
            ExecuteCopyBatch<Source, Result>(programs, source_batch, result + batch * output_size, output_size);
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map preparation exception"));
  }
}

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
void ExecuteUpdateBatch(
    const std::vector<layout::GeneratedRecordProgram>& programs,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t output_size, scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  static_assert(std::is_same_v<typename BaseMap::InputDtype, Result>);
  InitializeUpdateOutput<Result>(base, result, output_size);
  for (const auto& program : programs) {
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteUpdateRecord<Result, Alpha, Beta>(
          block, source, base, result, alpha, beta, source_map, base_map);
    });
  }
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
ffi::Future ExecuteUpdate(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    const scalar::Value<Alpha>* alpha, uint64_t alpha_count,
    const scalar::Value<Beta>* beta, uint64_t beta_count,
    SourceMap source_map, BaseMap base_map) {
  try {
    if ((alpha_count != 1 && alpha_count != batch_count) ||
        (beta_count != 1 && beta_count != batch_count)) {
      throw std::invalid_argument("update coefficients must be shared or one value per batch");
    }
    auto programs = layout::PrepareGeneratedRecords(records, sizeof(*source), sizeof(*result));
    if (batch_count == 1 && output_size != 0) {
      return ExecuteMapTasks(thread_pool, programs,
          [=] { InitializeUpdateOutput<Result>(base, result, output_size); },
          [=, alpha_value = *alpha, beta_value = *beta](const auto& record) {
            ExecuteUpdateRecord<Result, Alpha, Beta>(
                record, source, base, result, alpha_value, beta_value, source_map, base_map);
          });
    }
    return ExecuteBatchTasks(thread_pool, batch_count, output_size,
        [=, programs = std::move(programs)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            ExecuteUpdateBatch<Result, Alpha, Beta>(
                programs, source == nullptr ? nullptr : source + batch * source_size,
                base + batch * output_size, result + batch * output_size, output_size,
                alpha[alpha_count == 1 ? 0 : batch], beta[beta_count == 1 ? 0 : batch],
                source_map, base_map);
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map preparation exception"));
  }
}
