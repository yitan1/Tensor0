uint64_t ReductionWorkerCount(ffi::ThreadPool thread_pool,
                              const std::vector<layout::Record>& records) {
  uint64_t elements = 0;
  for (const auto& record : records) {
    if (!layout::CheckedAdd(elements, layout::ElementCount(record), &elements)) {
      throw std::invalid_argument("reduction work size overflows");
    }
  }
  return std::min(AvailableWorkerCount(thread_pool),
                  std::max<uint64_t>(1, elements / (UINT64_C(1) << 15)));
}

template <typename Accumulator>
inline constexpr bool kParallelReductionAccumulator =
    std::is_same_v<Accumulator, scalar::F32> || std::is_same_v<Accumulator, scalar::F64> ||
    std::is_same_v<Accumulator, scalar::C64> || std::is_same_v<Accumulator, scalar::C128>;

template <typename Accumulator>
ffi::Future ExecuteReductionPartials(
    ffi::ThreadPool thread_pool, std::vector<layout::Record> domains, uint64_t workers,
    scalar::Value<Accumulator>* result, uint64_t output_size, uint64_t output_index,
    std::function<void(const layout::Record&, scalar::Value<Accumulator>*)> execute) {
  using Value = scalar::Value<Accumulator>;
  struct alignas(64) Partial { Value value{}; };
  const auto count = domains.size();
  auto partials = std::make_shared<std::vector<Partial>>(count);
  return ExecuteTasks(thread_pool, count, workers,
      [domains = std::move(domains), partials, execute = std::move(execute)](uint64_t begin, uint64_t count) {
        for (uint64_t index = begin; index < begin + count; ++index) {
          execute(domains[index], &(*partials)[index].value);
        }
      }, [=] {
        Value sum{};
        for (const auto& partial : *partials) {
          sum = scalar::Add<Accumulator, Accumulator>(sum, partial.value);
        }
        std::fill_n(result, output_size, Value{});
        result[output_index] = sum;
      });
}

template <typename Accumulator, typename LeftOp, typename RightOp>
ffi::Future ExecuteDotTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* result,
    uint64_t left_size, uint64_t right_size, uint64_t batch_count,
    LeftOp left_op, RightOp right_op) {
  try {
    if constexpr (kParallelReductionAccumulator<Accumulator>) {
      if (batch_count == 1) {
        const auto workers = ReductionWorkerCount(thread_pool, records);
        if (workers > 1) {
          std::vector<layout::Record> domains;
          for (const auto& record : records) {
            if (layout::ElementCount(record) != 0) domains.push_back(record);
          }
          layout::SplitReductionDomains(&domains, workers, [](const auto& record, auto axis) {
            return std::min(std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis])),
                            std::max<uint64_t>(1, layout::AbsoluteStride(record.destination_strides[axis])));
          });
          return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(domains), workers,
              result, 1, 0, [=](const auto& record, auto* partial) {
                kernels::ExecuteDotRecord<Accumulator>(record, left, right, partial, left_op, right_op);
              });
        }
      }
    }
    return ExecuteBatchTasks(thread_pool, batch_count, std::max({left_size, right_size, UINT64_C(1)}),
        [=, &records](uint64_t begin, uint64_t count) {
          ExecuteDot<Accumulator>(records, left == nullptr ? nullptr : left + begin * left_size,
              right == nullptr ? nullptr : right + begin * right_size, result + begin,
              left_size, right_size, count, left_op, right_op);
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown dot task exception"));
  }
}

template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteReductionTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source) {
  try {
    if (batch_count == 1 && output_size != 0) {
      const auto workers = ReductionWorkerCount(thread_pool, records);
      if (workers > 1) {
        const auto execute_record = [=](const auto& record, auto* target) {
          bind_source(record.semantic_index, 0, [&](const auto& source_op) {
            kernels::ExecuteReductionRecord<Accumulator>(record, source, target, source_op);
          });
        };
        if constexpr (kParallelReductionAccumulator<Accumulator>) {
          int64_t destination = -1;
          bool single_destination = true;
          std::vector<layout::Record> domains;
          for (const auto& record : records) {
            if (layout::ElementCount(record) == 0) continue;
            if (destination == -1) destination = record.destination_offset;
            single_destination &= destination == record.destination_offset;
            for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
              single_destination &= record.shape[axis] <= 1 || record.destination_strides[axis] == 0;
            }
            domains.push_back(record);
            domains.back().destination_offset = 0;
          }
          if (single_destination && destination != -1) {
            layout::SplitReductionDomains(&domains, workers, [](const auto& record, auto axis) {
              return std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis]));
            });
            return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(domains), workers,
                result, output_size, destination, execute_record);
          }
        }
        auto tasks = layout::BuildReductionOutputTasks(records, workers);
        if (tasks.size() > 1) {
          const auto count = tasks.size();
          std::fill_n(result, output_size, scalar::Value<Accumulator>{});
          return ExecuteTasks(thread_pool, count, workers,
              [tasks = std::move(tasks), execute_record, result](uint64_t begin, uint64_t count) {
                for (uint64_t index = begin; index < begin + count; ++index) {
                  for (const auto& record : tasks[index]) execute_record(record, result);
                }
              });
        }
      }
    }
    return ExecuteBatchTasks(thread_pool, batch_count,
        output_size == 0 ? 0 : std::max(source_size, output_size),
        [=, &records](uint64_t begin, uint64_t count) {
          ExecuteReduction<Source, Accumulator>(records,
              source == nullptr ? nullptr : source + begin * source_size,
              result + begin * output_size, source_size, output_size, count,
              [&](std::size_t record, uint64_t batch, auto apply) {
                bind_source(record, begin + batch, apply);
              });
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown reduction task exception"));
  }
}
