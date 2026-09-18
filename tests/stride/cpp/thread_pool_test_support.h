#pragma once

#include <cassert>
#include <optional>
#include <thread>

namespace testing {

struct ThreadPool {
  XLA_FFI_Api api{};
  ffi::DiagnosticEngine diagnostic;
  int64_t threads;
  std::vector<std::pair<XLA_FFI_Task*, void*>> tasks;

  explicit ThreadPool(int64_t count = 1) : threads(count) {
    api.XLA_FFI_ThreadPool_NumThreads = [](XLA_FFI_ThreadPool_NumThreads_Args* args) -> XLA_FFI_Error* {
      *args->num_threads = reinterpret_cast<ThreadPool*>(args->ctx)->threads;
      return nullptr;
    };
    api.XLA_FFI_ThreadPool_Schedule = [](XLA_FFI_ThreadPool_Schedule_Args* args) -> XLA_FFI_Error* {
      reinterpret_cast<ThreadPool*>(args->ctx)->tasks.emplace_back(args->task, args->data);
      return nullptr;
    };
  }

  ffi::ThreadPool get() {
    return *ffi::CtxDecoding<ffi::ThreadPool>::Decode(
        &api, reinterpret_cast<XLA_FFI_ExecutionContext*>(this), diagnostic);
  }

  void run_one() {
    const auto task = tasks.back();
    tasks.pop_back();
    task.first(task.second);
  }

  void run_parallel() {
    std::vector<std::thread> workers;
    for (const auto& task : tasks) workers.emplace_back(task.first, task.second);
    tasks.clear();
    for (auto& worker : workers) worker.join();
  }

  ~ThreadPool() { assert(tasks.empty()); }
};

inline ffi::Error CompletedError(ffi::Future future) {
  bool ready = false;
  ffi::Error result = ffi::Error::Success();
  future.OnReady([&](const std::optional<ffi::Error>& error) {
    ready = true;
    if (error) result = *error;
  });
  assert(ready);
  return result;
}

}
