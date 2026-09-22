#pragma once

#include "../layout/blocking.h"
#include "xla/ffi/api/ffi.h"
#include <atomic>
#include <cstddef>
#include <new>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <utility>

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

// Raw storage types end at this scalar load/conversion boundary. Buffers are borrowed.
struct UpdateCoefficientReader {
  const void* data;
  uint64_t count;
  void (*load)(const void*, uint64_t, void*);
};

struct NoReduction {};

template <typename F, typename Op, typename Init>
struct Program {
  F f;
  [[no_unique_address]] Op op;
  Init init;
  template <typename... Args>
  Program(const Init& initialization, Args&&... args)
      : f(std::forward<Args>(args)...), init(initialization) {}
};

// Control-block code depends on storage layout, not the complete expression type.
template <std::size_t Size, std::size_t Alignment>
struct ProgramStorage {
  alignas(Alignment) std::byte data[Size];
  void (*destroy)(void*) = nullptr;

  ProgramStorage() noexcept {}  // Do not zero raw storage before constructing State.
  ~ProgramStorage() {
    if (destroy) destroy(data);
  }
};

template <typename State, typename... Args>
std::shared_ptr<const State> MakeProgramState(Args&&... args) {
  auto storage = std::make_shared<ProgramStorage<sizeof(State), alignof(State)>>();
  auto* state = std::construct_at(reinterpret_cast<State*>(storage->data), std::forward<Args>(args)...);
  // Install destruction only after construction succeeds; failed members unwind normally.
  storage->destroy = [](void* address) { std::destroy_at(std::launder(static_cast<State*>(address))); };
  return std::shared_ptr<const State>(std::move(storage), state);
}

template <typename F, typename Init>
using MapProgram = Program<F, NoReduction, Init>;

struct MapProgramEntry {
  void (*initialize)(const void*, uint64_t);
  std::function<void(const layout::Record&)> (*bind_record)(const void*, uint64_t);
};

template <typename Entry>
struct OwnedProgram {
  std::shared_ptr<const void> state;
  const Entry* entry;
};

using OwnedMapProgram = OwnedProgram<MapProgramEntry>;

ffi::Future ExecuteOwnedMapProgram(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    uint64_t source_item_size, uint64_t result_item_size, uint64_t output_size,
    uint64_t batch_count, const OwnedMapProgram& program);

uint64_t ReductionWorkerCount(ffi::ThreadPool thread_pool,
                              const std::vector<layout::Record>& records);

void ValidateReductionStorage(uint64_t source_size, uint64_t output_size, uint64_t batch_count,
                              uint64_t source_item_size, uint64_t result_item_size);

void ValidateDotStorage(uint64_t left_size, uint64_t right_size, uint64_t batch_count,
                        uint64_t left_item_size, uint64_t right_item_size, uint64_t result_item_size);

}

namespace tensor0::stride {

enum class ReductionKind { Reduce, Dot };
enum class ReductionMode { Batches, Outputs, Partials };

struct ReductionPlan {
  ReductionPlan() = default;
  ReductionPlan(ReductionPlan&&) noexcept = default;
  ~ReductionPlan();

  ReductionMode mode = ReductionMode::Batches;
  std::vector<layout::GeneratedRecordProgram> programs;
  std::vector<std::vector<layout::GeneratedRecordProgram>> tasks;
  uint64_t workers = 1;
  uint64_t output_index = 0;
  uint64_t work = 0;
};

ReductionPlan PrepareReductionPlan(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    ReductionKind kind, bool parallel_accumulator, uint64_t source_size,
    uint64_t right_size, uint64_t output_size, uint64_t batch_count,
    uint64_t source_item_size, uint64_t right_item_size, uint64_t result_item_size);

}

namespace tensor0::stride {

// Synchronous traversal only: neither the callback nor its context is retained.
void ExecuteGeneratedBlocks(
    const layout::GeneratedRecordProgram& program, const void* context,
    void (*execute)(const void*, const layout::Record&));

}
