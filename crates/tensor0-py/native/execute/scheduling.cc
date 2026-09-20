#include "scheduling.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
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

ffi::Future ExecuteUpdateTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    uint64_t source_item_size, uint64_t result_item_size,
    uint64_t output_size, uint64_t batch_count, uint64_t alpha_count, uint64_t beta_count,
    const void* base, void* result,
    std::function<std::function<void(const layout::Record&)>(uint64_t)> bind_batch) {
  if ((alpha_count != 1 && alpha_count != batch_count) ||
      (beta_count != 1 && beta_count != batch_count)) {
    throw std::invalid_argument("update coefficients must be shared or one value per batch");
  }
  auto programs = layout::PrepareGeneratedRecords(records, source_item_size, result_item_size);
  const uint64_t output_bytes = output_size * result_item_size;
  const auto initialize = [=](uint64_t batch) {
    if (output_bytes != 0 && result != base) {
      std::memcpy(static_cast<std::byte*>(result) + batch * output_bytes,
                  static_cast<const std::byte*>(base) + batch * output_bytes, output_bytes);
    }
  };
  if (batch_count == 1 && output_size != 0) {
    return ExecuteMapTasks(thread_pool, programs,
        [initialize] { initialize(0); }, bind_batch(0));
  }
  return ExecuteBatchTasks(thread_pool, batch_count, output_size,
      [programs = std::move(programs), initialize,
       bind_batch = std::move(bind_batch)](uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          const auto execute = bind_batch(batch);
          initialize(batch);
          for (const auto& program : programs) {
            layout::ForEachGeneratedBlock(program, std::cref(execute));
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
