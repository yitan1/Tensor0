ffi::Future ExecuteMapTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    uint64_t source_item_size, uint64_t result_item_size,
    std::function<void()> initialize,
    std::function<void(const layout::Record&)> execute) {
  try {
    uint64_t elements = 0;
    for (const auto& record : records) {
      if (!layout::CheckedAdd(elements, layout::ElementCount(record), &elements)) {
        throw std::invalid_argument("map work size overflows");
      }
    }
    const auto workers = std::min(
        AvailableWorkerCount(thread_pool),
        std::max<uint64_t>(1, elements / (UINT64_C(1) << 15)));
    std::vector<layout::GeneratedRecordProgram> programs;
    for (const auto& record : records) {
      if (layout::ElementCount(record) == 0) continue;
      const auto program = layout::CompileGeneratedRecord(
          record, true, true, source_item_size, result_item_size);
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

template <typename Source, typename Result>
ffi::Future ExecuteCopyBatches(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count) {
  if (batch_count == 1 && output_size != 0) {
    return ExecuteMapTasks(thread_pool, records, sizeof(*source), sizeof(*result),
        [=] { InitializeCopyOutput<Result>(result, output_size); },
        [=](const auto& record) { ExecuteCopyRecord<Source, Result>(record, source, result); });
  }
  return ExecuteBatchTasks(thread_pool, batch_count, output_size,
      [=, &records](uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
          ExecuteCopy<Source, Result>(records, source_batch, result + batch * output_size, output_size);
        }
      });
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
ffi::Future ExecuteUpdateTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    const scalar::Value<Alpha>* alpha, uint64_t alpha_count,
    const scalar::Value<Beta>* beta, uint64_t beta_count,
    SourceMap source_map, BaseMap base_map) {
  if (batch_count == 1 && output_size != 0) {
    return ExecuteMapTasks(thread_pool, records, sizeof(*source), sizeof(*result),
        [=] { InitializeUpdateOutput<Result>(base, result, output_size); },
        [=, alpha_value = *alpha, beta_value = *beta](const auto& record) {
          ExecuteUpdateRecord<Result, Alpha, Beta>(
              record, source, base, result, alpha_value, beta_value, source_map, base_map);
        });
  }
  return ExecuteBatchTasks(thread_pool, batch_count, output_size,
      [=, &records](uint64_t begin, uint64_t count) {
        ExecuteUpdateBatches<Result, Alpha, Beta>(
            records, source == nullptr ? nullptr : source + begin * source_size,
            base + begin * output_size, result + begin * output_size,
            source_size, output_size, count,
            alpha + (alpha_count == 1 ? 0 : begin), alpha_count == 1 ? 1 : count,
            beta + (beta_count == 1 ? 0 : begin), beta_count == 1 ? 1 : count,
            source_map, base_map);
      });
}
