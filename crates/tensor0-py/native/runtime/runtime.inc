#include <atomic>
#include <functional>
#include <memory>
#include <optional>

std::atomic<uint64_t> worker_limit{std::numeric_limits<uint64_t>::max()};

uint64_t AvailableWorkerCount(ffi::ThreadPool thread_pool) {
  return std::min(static_cast<uint64_t>(std::max<int64_t>(1, thread_pool.num_threads())),
                  std::max<uint64_t>(1, worker_limit.load(std::memory_order_relaxed)));
}

ffi::Future CompletedFuture(ffi::Error error = ffi::Error::Success()) {
  ffi::Promise completion;
  ffi::Future future(completion);
  if (error.success()) completion.SetAvailable();
  else completion.SetError(std::move(error));
  return future;
}

ffi::Future ExecuteTasks(
    ffi::ThreadPool thread_pool, uint64_t task_count, uint64_t worker_limit,
    std::function<void(uint64_t, uint64_t)> execute, std::function<void()> finish = {}) {
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

std::atomic<uint64_t> prepared_created_count{0};
std::atomic<uint64_t> prepared_destroyed_count{0};

struct PreparedState {
  static ffi::TypeId id;

  explicit PreparedState(descriptor::DecodedLayout value)
      : records(std::move(value.records)), source_size(value.source_size), output_size(value.output_size) {
    prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }

  PreparedState(const PreparedState&) = delete;
  PreparedState& operator=(const PreparedState&) = delete;

  ~PreparedState() {
    prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const std::vector<layout::Record> records;
  const uint64_t source_size;
  const uint64_t output_size;
};

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();
