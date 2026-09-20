#pragma once

#include "../layout/blocking.h"
#include "xla/ffi/api/ffi.h"
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace tensor0::stride {

namespace ffi = xla::ffi;

extern std::atomic<uint64_t> worker_limit;

uint64_t AvailableWorkerCount(ffi::ThreadPool thread_pool);

ffi::Future CompletedFuture(ffi::Error error = ffi::Error::Success());

ffi::Future ExecuteTasks(
    ffi::ThreadPool thread_pool, uint64_t task_count, uint64_t worker_limit,
    std::function<void(uint64_t, uint64_t)> execute, std::function<void()> finish = {});

ffi::Future ExecuteBatchTasks(
    ffi::ThreadPool thread_pool, uint64_t batch_count, uint64_t elements_per_batch,
    std::function<void(uint64_t, uint64_t)> execute);

ffi::Future ExecuteMapTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::GeneratedRecordProgram>& prepared,
    std::function<void()> initialize,
    std::function<void(const layout::Record&)> execute);

ffi::Future ExecuteUpdateTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    uint64_t source_item_size, uint64_t result_item_size,
    uint64_t output_size, uint64_t batch_count, uint64_t alpha_count, uint64_t beta_count,
    const void* base, void* result,
    std::function<std::function<void(const layout::Record&)>(uint64_t)> bind_batch);

uint64_t ReductionWorkerCount(ffi::ThreadPool thread_pool,
                              const std::vector<layout::Record>& records);

void ValidateReductionStorage(uint64_t source_size, uint64_t output_size, uint64_t batch_count,
                              uint64_t source_item_size, uint64_t result_item_size);

void ValidateDotStorage(uint64_t left_size, uint64_t right_size, uint64_t batch_count,
                        uint64_t left_item_size, uint64_t right_item_size, uint64_t result_item_size);

}
