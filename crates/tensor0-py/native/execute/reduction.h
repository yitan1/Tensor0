#pragma once

#include "reduction_partials.h"
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

// Share planning and scheduling across source types and coefficient binders.
template <typename Accumulator>
ffi::Future ExecuteReductionTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, uint64_t source_item_size,
    std::function<void(const layout::GeneratedRecordProgram&, uint64_t,
                       scalar::Value<Accumulator>*)> execute_program) {
  ValidateReductionStorage(source_size, output_size, batch_count, source_item_size, sizeof(*result));
  auto programs = layout::PrepareGeneratedRecords(records, source_item_size, sizeof(*result), false);
  if (batch_count == 1 && output_size != 0) {
    const auto workers = ReductionWorkerCount(thread_pool, records);
    if (workers > 1) {
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
              result, output_size, destination,
              [execute_program = std::move(execute_program)](const auto& program, auto* target) {
                execute_program(program, 0, target);
              });
        }
      }
      auto tasks = layout::BuildReductionOutputTasks(programs, workers);
      if (tasks.size() > 1) {
        const auto count = tasks.size();
        std::fill_n(result, output_size, scalar::Value<Accumulator>{});
        return ExecuteTasks(thread_pool, count, workers,
            [tasks = std::move(tasks), execute_program = std::move(execute_program), result]
            (uint64_t begin, uint64_t count) {
              for (uint64_t index = begin; index < begin + count; ++index) {
                for (const auto& program : tasks[index]) execute_program(program, 0, result);
              }
            });
      }
    }
  }
  return ExecuteBatchTasks(thread_pool, batch_count,
      output_size == 0 ? 0 : std::max(source_size, output_size),
      [programs = std::move(programs), execute_program = std::move(execute_program), result, output_size]
      (uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          auto* target = result + batch * output_size;
          std::fill_n(target, output_size, scalar::Value<Accumulator>{});
          for (const auto& program : programs) execute_program(program, batch, target);
        }
      });
}

template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteReduction(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source) {
  try {
    return ExecuteReductionTasks<Accumulator>(thread_pool, records, result,
        source_size, output_size, batch_count, sizeof(*source),
        [=, bind_source = std::move(bind_source)]
        (const layout::GeneratedRecordProgram& program, uint64_t batch, scalar::Value<Accumulator>* target) {
          const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
          bind_source(program.record.semantic_index, batch, [&](const auto& source_op) {
            using SourceOp = std::remove_cvref_t<decltype(source_op)>;
            static_assert(std::is_same_v<typename SourceOp::InputDtype, Source>);
            layout::ForEachGeneratedBlock(program, [&](const auto& block) {
              kernels::ExecuteReductionRecord<Accumulator>(block, source_batch, target, source_op);
            });
          });
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown reduction task exception"));
  }
}

}
