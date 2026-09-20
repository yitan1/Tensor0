#pragma once

#include "scheduling.h"
#include "../kernels/dispatch.h"
#include "../layout/blocking.h"
#include "../numeric/expression.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor0::stride {

// Initialize the entire batch, never an individual record or parallel subdomain.
template <typename Result>
void InitializeCopyOutput(scalar::Value<Result>* result, uint64_t output_size) {
  if (output_size != 0) std::fill_n(result, output_size, scalar::Value<Result>{});
}

template <typename Source, typename Result = Source>
void ExecuteCopyRecord(const layout::Record& record, const scalar::Value<Source>* source,
                       scalar::Value<Result>* result) {
  if constexpr (std::is_same_v<Source, Result>) {
    kernels::ExecuteMapRecord(record, source, result, expression::Identity<Source>{});
  } else {
    kernels::ExecuteMapRecord(
        record, source, result, expression::Cast<Result, expression::Identity<Source>>{});
  }
}

// Synchronously execute one complete batch, including output initialization.
template <typename Source, typename Result = Source>
void ExecuteCopyBatch(const std::vector<layout::GeneratedRecordProgram>& programs, const scalar::Value<Source>* source,
                 scalar::Value<Result>* result, uint64_t output_size) {
  InitializeCopyOutput<Result>(result, output_size);
  for (const auto& program : programs) {
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteCopyRecord<Source, Result>(block, source, result);
    });
  }
}

template <typename Source, typename Result>
ffi::Future ExecuteCopy(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count) {
  try {
    auto programs = layout::PrepareGeneratedRecords(records, sizeof(*source), sizeof(*result));
    if (batch_count == 1 && output_size != 0) {
      return ExecuteMapTasks(thread_pool, programs,
          [=] { InitializeCopyOutput<Result>(result, output_size); },
          [=](const auto& record) { ExecuteCopyRecord<Source, Result>(record, source, result); });
    }
    return ExecuteBatchTasks(thread_pool, batch_count, output_size,
        [=, programs = std::move(programs)](uint64_t begin, uint64_t count) {
          for (uint64_t batch = begin; batch < begin + count; ++batch) {
            const auto* source_batch = source == nullptr ? nullptr : source + batch * source_size;
            ExecuteCopyBatch<Source, Result>(programs, source_batch, result + batch * output_size, output_size);
          }
        });
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map preparation exception"));
  }
}

}
