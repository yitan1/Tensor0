#pragma once

#include "reduction_program.h"
#include "../kernels/generic.h"
#include "../layout/blocking.h"
#include "../numeric/scalar.h"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor0::stride {

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

// Producer-proven injective, cross-record disjoint output maps. Only task
// scheduling differs from accumulation; coefficients and writeback are unchanged.
template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteDisjointReduction(
    ffi::ThreadPool thread_pool,
    std::shared_ptr<const std::vector<layout::GeneratedRecordProgram>> programs,
    uint64_t work, const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source) {
  if (batch_count == 0 || output_size == 0) return CompletedFuture();
  if (batch_count != 1) {
    // Keep the existing independent-batch schedule and initialize each batch once.
    return ExecuteBatchTasks(thread_pool, batch_count, std::max(source_size, output_size),
        [=, bind_source = std::move(bind_source)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            ExecuteReductionBatch<Source, Accumulator>(
                *programs, source == nullptr ? nullptr : source + batch * source_size,
                result + batch * output_size, output_size,
                [&](std::size_t record, auto apply) { bind_source(record, batch, apply); });
          }
        });
  }
  const auto workers = std::min(AvailableWorkerCount(thread_pool),
      std::max<uint64_t>(1, work / (UINT64_C(1) << 15)));
  if (workers == 1 || programs->empty()) {
    ExecuteReductionBatch<Source, Accumulator>(*programs, source, result, output_size,
        [&](std::size_t record, auto apply) { bind_source(record, 0, apply); });
    return CompletedFuture();
  }
  if (programs->size() == 1) {
    // A single large injective map can use the existing bounded subdomain
    // splitter. No global output geometry or cross-record copies are needed.
    const auto& prepared = *programs;
    return ExecuteMapTasks(thread_pool, prepared,
        [=] { std::fill_n(result, output_size, scalar::Value<Accumulator>{}); },
        [=, bind_source = std::move(bind_source)](const layout::Record& block) {
          bind_source(block.semantic_index, 0, [&](const auto& source_op) {
            kernels::ExecuteReductionRecord<Accumulator>(block, source, result, source_op);
          });
        });
  }
  std::fill_n(result, output_size, scalar::Value<Accumulator>{});
  const auto count = programs->size();
  // ExecuteTasks creates at most `workers` tasks; each owns a contiguous range
  // of shared immutable programs, not a deep copy or one task per record.
  return ExecuteTasks(thread_pool, count, workers,
      [programs = std::move(programs), source, result, bind_source = std::move(bind_source)]
      (uint64_t begin, uint64_t count) {
        for (uint64_t index = begin; index < begin + count; ++index) {
          const auto& program = (*programs)[index];
          bind_source(program.record.semantic_index, 0, [&](const auto& source_op) {
            layout::ForEachGeneratedBlock(program, [&](const auto& block) {
              kernels::ExecuteReductionRecord<Accumulator>(block, source, result, source_op);
            });
          });
        }
      });
}

template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteReduction(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source,
    std::shared_ptr<const std::vector<layout::GeneratedRecordProgram>> shared_programs = {}) {
  try {
    auto plan = PrepareReductionPlan(thread_pool, records, ReductionKind::Reduce,
        kParallelReductionAccumulator<Accumulator>, source_size, 0, output_size, batch_count,
        sizeof(*source), 0, sizeof(*result), std::move(shared_programs));
    auto program = MakeReductionProgram<Accumulator>(
        [=, bind_source = std::move(bind_source)]
        (const layout::GeneratedRecordProgram& program, uint64_t batch, scalar::Value<Accumulator>* target,
         std::type_identity<Accumulator>) {
          const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
          bind_source(program.record.semantic_index, batch, [&](const auto& source_op) {
            using SourceOp = std::remove_cvref_t<decltype(source_op)>;
            static_assert(std::is_same_v<typename SourceOp::InputDtype, Source>);
            ExecuteReductionBlocks(program, [&](const auto& block) {
              kernels::ExecuteReductionRecord<Accumulator>(block, source_batch, target, source_op);
            });
          });
        });
    return ExecuteOwnedReductionProgram<Accumulator>(thread_pool, std::move(plan),
        result, output_size, batch_count, std::move(program));
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown reduction task exception"));
  }
}

}
