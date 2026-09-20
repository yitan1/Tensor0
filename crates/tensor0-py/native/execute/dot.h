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

// Scheduling depends on accumulator storage, not on input expressions.
template <typename Accumulator>
ffi::Future ExecuteDotTasks(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    scalar::Value<Accumulator>* result,
    uint64_t left_size, uint64_t right_size, uint64_t batch_count,
    uint64_t left_item_size, uint64_t right_item_size,
    std::function<void(const layout::Record&, uint64_t, scalar::Value<Accumulator>*)> execute) {
  ValidateDotStorage(left_size, right_size, batch_count,
                     left_item_size, right_item_size, sizeof(*result));
  auto programs = layout::PrepareGeneratedRecords(records, left_item_size, right_item_size, false);
  if constexpr (kParallelReductionAccumulator<Accumulator>) {
    if (batch_count == 1) {
      const auto workers = ReductionWorkerCount(thread_pool, records);
      if (workers > 1) {
        layout::SplitReductionDomains(&programs, workers, [](const auto& record, auto axis) {
          return std::min(std::max<uint64_t>(1, layout::AbsoluteStride(record.source_strides[axis])),
                          std::max<uint64_t>(1, layout::AbsoluteStride(record.destination_strides[axis])));
        });
        return ExecuteReductionPartials<Accumulator>(thread_pool, std::move(programs), workers,
            result, 1, 0, [execute = std::move(execute)](const auto& program, auto* partial) {
              layout::ForEachGeneratedBlock(program, [&](const auto& block) {
                execute(block, 0, partial);
              });
            });
      }
    }
  }
  return ExecuteBatchTasks(thread_pool, batch_count, std::max({left_size, right_size, UINT64_C(1)}),
      [programs = std::move(programs), execute = std::move(execute), result]
      (uint64_t begin, uint64_t count) {
        for (uint64_t batch = begin; batch < begin + count; ++batch) {
          auto* target = result + batch;
          *target = {};
          for (const auto& program : programs) {
            layout::ForEachGeneratedBlock(program, [&](const auto& block) {
              execute(block, batch, target);
            });
          }
        }
      });
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
    return ExecuteDotTasks<Accumulator>(thread_pool, records, result,
        left_size, right_size, batch_count, sizeof(*left), sizeof(*right),
        [=](const layout::Record& block, uint64_t batch, scalar::Value<Accumulator>* target) {
          kernels::ExecuteDotRecord<Accumulator>(block,
              left == nullptr ? nullptr : left + batch * left_size,
              right == nullptr ? nullptr : right + batch * right_size,
              target, left_op, right_op);
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown dot task exception"));
  }
}

}
