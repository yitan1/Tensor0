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
    ffi::ThreadPool thread_pool, std::vector<layout::GeneratedRecordProgram> domains, uint64_t workers,
    scalar::Value<Accumulator>* result, uint64_t output_size, uint64_t output_index,
    std::function<void(const layout::GeneratedRecordProgram&, scalar::Value<Accumulator>*)> execute) {
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

template <typename Source, typename Accumulator>
void ValidateReductionStorage(uint64_t source_size, uint64_t output_size, uint64_t batch_count) {
  uint64_t source_elements = 0;
  uint64_t output_elements = 0;
  const auto maximum_bytes = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!layout::CheckedMultiply(source_size, batch_count, &source_elements) ||
      !layout::CheckedMultiply(output_size, batch_count, &output_elements) ||
      source_elements > maximum_bytes / sizeof(scalar::Value<Source>) ||
      output_elements > maximum_bytes / sizeof(scalar::Value<Accumulator>)) {
    throw std::invalid_argument("reduction batch storage exceeds address range");
  }
}

// Synchronously initialize and execute one complete batch in record order.
template <typename Source, typename Accumulator, typename BindSource>
void ExecuteReductionBatch(
    const std::vector<layout::GeneratedRecordProgram>& programs,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* accumulator,
    uint64_t output_size, BindSource bind_source) {
  if (output_size == 0) return;
  std::fill_n(accumulator, output_size, scalar::Value<Accumulator>{});
  for (const auto& program : programs) {
    bind_source(program.record.semantic_index, [&](const auto& source_op) {
      using SourceOp = std::remove_cvref_t<decltype(source_op)>;
      static_assert(std::is_same_v<typename SourceOp::InputDtype, Source>);
      layout::ForEachGeneratedBlock(program, [&](const auto& block) {
        kernels::ExecuteReductionRecord<Accumulator>(block, source, accumulator, source_op);
      });
    });
  }
}

template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteReduction(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source) {
  try {
    ValidateReductionStorage<Source, Accumulator>(source_size, output_size, batch_count);
    auto programs = layout::PrepareGeneratedRecords(records, sizeof(*source), sizeof(*result), false);
    if (batch_count == 1 && output_size != 0) {
      const auto workers = ReductionWorkerCount(thread_pool, records);
      if (workers > 1) {
        const auto execute_program = [=](const auto& program, auto* target) {
          bind_source(program.record.semantic_index, 0, [&](const auto& source_op) {
            layout::ForEachGeneratedBlock(program, [&](const auto& block) {
              kernels::ExecuteReductionRecord<Accumulator>(block, source, target, source_op);
            });
          });
        };
        if constexpr (kParallelReductionAccumulator<Accumulator>) {
          int64_t destination = -1;
          bool single_destination = true;
          for (const auto& program : programs) {
            const auto& record = program.record;
            if (destination == -1) destination = record.destination_offset;
            single_destination &= destination == record.destination_offset;
            for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
              single_destination &= record.shape[axis] <= 1 || record.destination_strides[axis] == 0;
            }
          }
          if (single_destination && destination != -1) {
            for (auto& program : programs) program.record.destination_offset = 0;
            layout::SplitReductionDomains(&programs, workers, [](const auto& record, auto axis) {
              return std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis]));
            });
            return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(programs), workers,
                result, output_size, destination, execute_program);
          }
        }
        auto tasks = layout::BuildReductionOutputTasks(programs, workers);
        if (tasks.size() > 1) {
          const auto count = tasks.size();
          std::fill_n(result, output_size, scalar::Value<Accumulator>{});
          return ExecuteTasks(thread_pool, count, workers,
              [tasks = std::move(tasks), execute_program, result](uint64_t begin, uint64_t count) {
                for (uint64_t index = begin; index < begin + count; ++index) {
                  for (const auto& program : tasks[index]) execute_program(program, result);
                }
              });
        }
      }
    }
    return ExecuteBatchTasks(thread_pool, batch_count,
        output_size == 0 ? 0 : std::max(source_size, output_size),
        [=, programs = std::move(programs)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            ExecuteReductionBatch<Source, Accumulator>(programs,
                source == nullptr ? nullptr : source + batch * source_size,
                result + batch * output_size, output_size,
                [&](std::size_t record, auto apply) { bind_source(record, batch, apply); });
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown reduction task exception"));
  }
}

template <typename Accumulator, typename LeftOp, typename RightOp>
void ValidateDotStorage(uint64_t left_size, uint64_t right_size, uint64_t batch_count) {
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
}

// Synchronously initialize and execute one complete batch in record order.
template <typename Accumulator, typename LeftOp, typename RightOp>
void ExecuteDotBatch(
    const std::vector<layout::GeneratedRecordProgram>& programs,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* result,
    const LeftOp& left_op, const RightOp& right_op) {
  *result = {};
  for (const auto& program : programs) {
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      kernels::ExecuteDotRecord<Accumulator>(block, left, right, result, left_op, right_op);
    });
  }
}

template <typename Accumulator, typename LeftOp, typename RightOp>
ffi::Future ExecuteDot(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* result,
    uint64_t left_size, uint64_t right_size, uint64_t batch_count,
    LeftOp left_op, RightOp right_op) {
  try {
    ValidateDotStorage<Accumulator, LeftOp, RightOp>(left_size, right_size, batch_count);
    auto programs = layout::PrepareGeneratedRecords(records, sizeof(*left), sizeof(*right), false);
    if constexpr (kParallelReductionAccumulator<Accumulator>) {
      if (batch_count == 1) {
        const auto workers = ReductionWorkerCount(thread_pool, records);
        if (workers > 1) {
          layout::SplitReductionDomains(&programs, workers, [](const auto& record, auto axis) {
            return std::min(std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis])),
                            std::max<uint64_t>(1, layout::AbsoluteStride(record.destination_strides[axis])));
          });
          return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(programs), workers,
              result, 1, 0, [=](const auto& program, auto* partial) {
                layout::ForEachGeneratedBlock(program, [&](const auto& block) {
                  kernels::ExecuteDotRecord<Accumulator>(block, left, right, partial, left_op, right_op);
                });
              });
        }
      }
    }
    return ExecuteBatchTasks(thread_pool, batch_count, std::max({left_size, right_size, UINT64_C(1)}),
        [=, programs = std::move(programs)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            ExecuteDotBatch<Accumulator>(programs,
                left == nullptr ? nullptr : left + batch * left_size,
                right == nullptr ? nullptr : right + batch * right_size,
                result + batch, left_op, right_op);
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown dot task exception"));
  }
}
