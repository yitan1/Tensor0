#include "scheduling.h"
#include "reduction_program.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensor0::stride {

std::atomic<uint64_t> worker_limit{std::numeric_limits<uint64_t>::max()};

uint64_t AvailableWorkerCount(ffi::ThreadPool thread_pool) {
  return std::min(static_cast<uint64_t>(std::max<int64_t>(1, thread_pool.num_threads())),
                  std::max<uint64_t>(1, worker_limit.load(std::memory_order_relaxed)));
}

ffi::Future CompletedFuture(ffi::Error error) {
  ffi::Promise completion;
  ffi::Future future(completion);
  if (error.success()) completion.SetAvailable();
  else completion.SetError(std::move(error));
  return future;
}

ffi::Future ExecuteTasks(
    ffi::ThreadPool thread_pool, uint64_t task_count, uint64_t worker_limit,
    std::function<void(uint64_t, uint64_t)> execute, std::function<void()> finish) {
  try {
    if (task_count == 0) {
      if (finish) finish();
      return CompletedFuture();
    }
    const uint64_t workers = std::min({task_count, AvailableWorkerCount(thread_pool),
        std::max<uint64_t>(1, worker_limit)});
    if (workers == 1) {
      execute(0, task_count);
      if (finish) finish();
      return CompletedFuture();
    }
    auto task = std::make_shared<const std::function<void(uint64_t, uint64_t)>>(std::move(execute));
    ffi::CountDownPromise completion(static_cast<int64_t>(workers));
    ffi::Future future(completion);
    std::optional<ffi::Future> finished;
    if (finish) {
      ffi::Promise final_completion;
      finished.emplace(final_completion);
      future.OnReady([finish = std::move(finish), final_completion](const std::optional<ffi::Error>& error) mutable {
        if (error) {
          final_completion.SetError(*error);
          return;
        }
        try {
          finish();
          final_completion.SetAvailable();
        } catch (const std::exception& failure) {
          final_completion.SetError(ffi::Error::Internal(std::string("tensor0-native completion: ") + failure.what()));
        } catch (...) {
          final_completion.SetError(ffi::Error::Internal("tensor0-native: unknown completion exception"));
        }
      });
    }
    uint64_t scheduled = 0;
    try {
      for (; scheduled < workers; ++scheduled) {
        const uint64_t count = task_count / workers + (scheduled < task_count % workers);
        const uint64_t begin = scheduled * (task_count / workers) +
            std::min(scheduled, task_count % workers);
        thread_pool.Schedule([task, completion, begin, count]() mutable {
          ffi::Error error = ffi::Error::Success();
          try {
            (*task)(begin, count);
          } catch (const std::exception& failure) {
            error = ffi::Error::Internal(std::string("tensor0-native worker: ") + failure.what());
          } catch (...) {
            error = ffi::Error::Internal("tensor0-native: unknown worker exception");
          }
          completion.CountDown(std::move(error));
        });
      }
    } catch (...) {
      completion.CountDown(workers - scheduled,
          ffi::Error::Internal("tensor0-native: failed to schedule task"));
    }
    return finished ? std::move(*finished) : std::move(future);
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown task execution exception"));
  }
}

ffi::Future ExecuteBatchTasks(
    ffi::ThreadPool thread_pool, uint64_t batch_count, uint64_t elements_per_batch,
    std::function<void(uint64_t, uint64_t)> execute) {
  if (batch_count == 0 || elements_per_batch == 0) return CompletedFuture();
  constexpr uint64_t kMinimumWorkerElements = UINT64_C(1) << 15;
  uint64_t elements = 0;
  if (!layout::CheckedMultiply(batch_count, elements_per_batch, &elements)) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: batch work size overflows"));
  }
  return ExecuteTasks(thread_pool, batch_count,
                      std::max<uint64_t>(1, elements / kMinimumWorkerElements), std::move(execute));
}

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

ffi::Future ExecuteOwnedMapProgram(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    uint64_t source_item_size, uint64_t result_item_size, uint64_t output_size,
    uint64_t batch_count, const OwnedMapProgram& program) {
  auto prepared = layout::PrepareGeneratedRecords(records, source_item_size, result_item_size);
  if (batch_count == 1 && output_size != 0) {
    auto execute = program.entry->bind_record(program.state.get(), 0);
    return ExecuteMapTasks(thread_pool, prepared,
        [&program] { program.entry->initialize(program.state.get(), 0); }, std::move(execute));
  }
  return ExecuteBatchTasks(thread_pool, batch_count, output_size,
      [program, prepared = std::move(prepared)](uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          const auto execute = program.entry->bind_record(program.state.get(), batch);
          program.entry->initialize(program.state.get(), batch);
          for (const auto& record : prepared) {
            layout::ForEachGeneratedBlock(record, std::cref(execute));
          }
        }
      });
}

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

void ValidateReductionStorage(uint64_t source_size, uint64_t output_size, uint64_t batch_count,
                              uint64_t source_item_size, uint64_t result_item_size) {
  uint64_t source_elements = 0;
  uint64_t output_elements = 0;
  const auto maximum_bytes = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!layout::CheckedMultiply(source_size, batch_count, &source_elements) ||
      !layout::CheckedMultiply(output_size, batch_count, &output_elements) ||
      source_elements > maximum_bytes / source_item_size ||
      output_elements > maximum_bytes / result_item_size) {
    throw std::invalid_argument("reduction batch storage exceeds address range");
  }
}

void ValidateDotStorage(uint64_t left_size, uint64_t right_size, uint64_t batch_count,
                        uint64_t left_item_size, uint64_t right_item_size, uint64_t result_item_size) {
  uint64_t left_elements = 0;
  uint64_t right_elements = 0;
  const auto maximum_bytes = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (!layout::CheckedMultiply(left_size, batch_count, &left_elements) ||
      !layout::CheckedMultiply(right_size, batch_count, &right_elements) ||
      left_elements > maximum_bytes / left_item_size ||
      right_elements > maximum_bytes / right_item_size ||
      batch_count > maximum_bytes / result_item_size) {
    throw std::invalid_argument("dot batch storage exceeds address range");
  }
}

}

extern "C" uint64_t Tensor0StrideGetWorkerLimit() {
  return tensor0::stride::worker_limit.load(std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetWorkerLimit(uint64_t limit) {
  tensor0::stride::worker_limit.store(limit, std::memory_order_relaxed);
}

namespace tensor0::stride {

// Keep nested record/vector cleanup out of expression-specific entry points.
ReductionPlan::~ReductionPlan() = default;

ReductionPlan PrepareReductionPlan(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    ReductionKind kind, bool parallel_accumulator, uint64_t source_size,
    uint64_t right_size, uint64_t output_size, uint64_t batch_count,
    uint64_t source_item_size, uint64_t right_item_size, uint64_t result_item_size) {
  const bool dot = kind == ReductionKind::Dot;
  if (dot) {
    ValidateDotStorage(source_size, right_size, batch_count,
                       source_item_size, right_item_size, result_item_size);
  } else {
    ValidateReductionStorage(source_size, output_size, batch_count,
                             source_item_size, result_item_size);
  }
  ReductionPlan plan;
  plan.programs = layout::PrepareGeneratedRecords(
      records, source_item_size, dot ? right_item_size : result_item_size, false);
  plan.work = dot ? std::max({source_size, right_size, UINT64_C(1)})
                  : output_size == 0 ? 0 : std::max(source_size, output_size);
  if (batch_count != 1 || output_size == 0 || (dot && !parallel_accumulator)) return plan;
  plan.workers = ReductionWorkerCount(thread_pool, records);
  if (plan.workers <= 1) return plan;
  int64_t destination = dot ? 0 : -1;
  bool single_destination = true;
  if (parallel_accumulator) {
    if (!dot) {
      for (const auto& program : plan.programs) {
        const auto& record = program.record;
        if (destination == -1) destination = record.destination_offset;
        single_destination &= destination == record.destination_offset;
        for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
          single_destination &= record.shape[axis] <= 1 || record.destination_strides[axis] == 0;
        }
      }
    }
    if (single_destination && destination != -1) {
      if (!dot) {
        for (auto& program : plan.programs) program.record.destination_offset = 0;
      }
      layout::SplitReductionDomains(&plan.programs, plan.workers,
          [dot](const auto& record, auto axis) {
            const auto source = std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis]));
            return dot ? std::min(source, std::max<uint64_t>(1,
                layout::AbsoluteStride(record.destination_strides[axis]))) : source;
          });
      plan.mode = ReductionMode::Partials;
      plan.output_index = destination;
      return plan;
    }
  }
  if (!dot) {
    plan.tasks = layout::BuildReductionOutputTasks(plan.programs, plan.workers);
    if (plan.tasks.size() > 1) plan.mode = ReductionMode::Outputs;
  }
  return plan;
}

}

namespace tensor0::stride {

void ExecuteGeneratedBlocks(
    const layout::GeneratedRecordProgram& program, const void* context,
    void (*execute)(const void*, const layout::Record&)) {
  layout::ForEachGeneratedBlock(program, [=](const layout::Record& block) {
    execute(context, block);
  });
}

}

namespace tensor0::stride {

// Only accumulator storage specializes task management. F is restored at a record boundary.
template <typename Accumulator>
ffi::Future ExecuteOwnedReductionProgram(
    ffi::ThreadPool thread_pool, ReductionPlan plan,
    scalar::Value<Accumulator>* result, uint64_t output_size, uint64_t batch_count,
    OwnedProgram<ReductionProgramEntry<Accumulator>> program) {
  if constexpr (kParallelReductionAccumulator<Accumulator>) {
    if (plan.mode == ReductionMode::Partials) {
      // Local zero identities and final output writes belong to the partials executor.
      return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(plan.programs),
          plan.workers, result, output_size, plan.output_index,
          [program = std::move(program)](const auto& record, auto* partial) {
            program.entry->execute(program.state.get(), record, 0, partial);
          });
    }
  }
  if (plan.mode == ReductionMode::Outputs) {
    const auto count = plan.tasks.size();
    program.entry->initialize(program.state.get(), result, output_size);
    return ExecuteTasks(thread_pool, count, plan.workers,
        [tasks = std::move(plan.tasks), program = std::move(program), result]
        (uint64_t begin, uint64_t count) {
          for (uint64_t index = begin; index < begin + count; ++index) {
            for (const auto& record : tasks[index]) {
              program.entry->execute(program.state.get(), record, 0, result);
            }
          }
        });
  }
  return ExecuteBatchTasks(thread_pool, batch_count, plan.work,
      [records = std::move(plan.programs), program = std::move(program), result, output_size]
      (uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          auto* target = result + batch * output_size;
          program.entry->initialize(program.state.get(), target, output_size);
          for (const auto& record : records) {
            program.entry->execute(program.state.get(), record, batch, target);
          }
        }
      });
}

template ffi::Future ExecuteOwnedReductionProgram<scalar::Pred>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::Pred>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::Pred>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::S8>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::S8>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::S8>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::S16>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::S16>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::S16>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::S32>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::S32>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::S32>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::S64>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::S64>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::S64>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::U8>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::U8>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::U8>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::U16>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::U16>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::U16>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::U32>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::U32>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::U32>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::U64>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::U64>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::U64>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::F16>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::F16>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::F16>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::BF16>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::BF16>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::BF16>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::F32>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::F32>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::F32>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::F64>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::F64>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::F64>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::C64>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::C64>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::C64>>);
template ffi::Future ExecuteOwnedReductionProgram<scalar::C128>(
    ffi::ThreadPool, ReductionPlan, scalar::Value<scalar::C128>*, uint64_t, uint64_t,
    OwnedProgram<ReductionProgramEntry<scalar::C128>>);

}
