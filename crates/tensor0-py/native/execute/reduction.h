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

template <typename Source, typename Accumulator, typename BindSource>
ffi::Future ExecuteReduction(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Accumulator>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count, BindSource bind_source) {
  try {
    auto plan = PrepareReductionPlan(thread_pool, records, ReductionKind::Reduce,
        kParallelReductionAccumulator<Accumulator>, source_size, 0, output_size, batch_count,
        sizeof(*source), 0, sizeof(*result));
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
