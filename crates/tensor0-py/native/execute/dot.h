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

// Input mapping state does not depend on the output accumulator dtype.
template <typename LeftOp, typename RightOp>
struct DotInput {
  const scalar::Value<typename LeftOp::InputDtype>* left;
  uint64_t left_size;
  const scalar::Value<typename RightOp::InputDtype>* right;
  uint64_t right_size;
  LeftOp left_op;
  RightOp right_op;

  template <typename Accumulator>
  void operator()(const layout::GeneratedRecordProgram& generated, uint64_t batch,
                  scalar::Value<Accumulator>* target, std::type_identity<Accumulator>) const {
    const auto* left_batch = left == nullptr ? nullptr : left + batch * left_size;
    const auto* right_batch = right == nullptr ? nullptr : right + batch * right_size;
    ExecuteReductionBlocks(generated, [&](const auto& block) {
      kernels::ExecuteDotRecord<Accumulator>(block, left_batch, right_batch,
                                             target, left_op, right_op);
    });
  }
};

template <typename Accumulator, typename LeftOp, typename RightOp>
ffi::Future ExecuteDot(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename LeftOp::InputDtype>* left,
    const scalar::Value<typename RightOp::InputDtype>* right,
    scalar::Value<Accumulator>* result,
    uint64_t left_size, uint64_t right_size, uint64_t batch_count,
    LeftOp left_op, RightOp right_op) {
  try {
    auto plan = PrepareReductionPlan(thread_pool, records, ReductionKind::Dot,
        kParallelReductionAccumulator<Accumulator>, left_size, right_size, 1, batch_count,
        sizeof(*left), sizeof(*right), sizeof(*result));
    auto program = MakeReductionProgram<Accumulator>(
        DotInput<LeftOp, RightOp>{left, left_size, right, right_size, left_op, right_op});
    return ExecuteOwnedReductionProgram<Accumulator>(thread_pool, std::move(plan),
        result, 1, batch_count, std::move(program));
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown dot task exception"));
  }
}

}
