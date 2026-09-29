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

// A single dense destination permutation needs no prefill. Check the affine
// span directly, including negative strides; do not assume the direct native
// caller supplied a DecodeLayout-validated record.
inline bool CopyCoversOutput(const std::vector<layout::Record>& records, uint64_t size) {
  if (records.size() != 1 || size == 0) return false;
  const auto& record = records.front();
  if (record.destination_strides.size() != record.shape.size() ||
      record.destination_offset < 0) return false;
  uint64_t span = 1;
  uint64_t minimum = static_cast<uint64_t>(record.destination_offset);
  std::size_t remaining = 0;
  for (auto extent : record.shape) {
    if (extent == 0) return false;
    remaining += extent > 1;
  }
  while (remaining-- != 0) {
    std::size_t axis = record.shape.size();
    for (std::size_t i = 0; i < record.shape.size(); ++i) {
      if (record.shape[i] > 1 && layout::AbsoluteStride(record.destination_strides[i]) == span) {
        axis = i;
        break;
      }
    }
    if (axis == record.shape.size() || record.shape[axis] > size / span) return false;
    const uint64_t width = (record.shape[axis] - 1) * span;
    if (record.destination_strides[axis] < 0) {
      if (minimum < width) return false;
      minimum -= width;
    }
    span *= record.shape[axis];
  }
  return minimum == 0 && span == size;
}

template <typename Result>
struct CopyInit {
  scalar::Value<Result>* result;
  uint64_t size;
  bool covers_output;
  void operator()(uint64_t batch) const {
    if (!covers_output) InitializeCopyOutput<Result>(result + batch * size, size);
  }
};

template <typename Source>
struct CopyF {
  const scalar::Value<Source>* source;
  uint64_t size;
  CopyF(const scalar::Value<Source>* input, uint64_t count) : source(input), size(count) {}
};

template <typename Source, typename Result>
using CopyProgram = MapProgram<CopyF<Source>, CopyInit<Result>>;

template <typename Source, typename Result>
const MapProgramEntry& CopyProgramEntry() {
  using Program = CopyProgram<Source, Result>;
  static const MapProgramEntry entry{
    [](const void* state, uint64_t batch) { static_cast<const Program*>(state)->init(batch); },
    [](const void* state, uint64_t batch) -> std::function<void(const layout::Record&)> {
      const auto& p = *static_cast<const Program*>(state);
      const auto* source = p.f.source == nullptr ? nullptr : p.f.source + batch * p.f.size;
      auto* result = p.init.result + batch * p.init.size;
      return [source, result](const layout::Record& record) {
        ExecuteCopyRecord<Source, Result>(record, source, result);
      };
    }
  };
  return entry;
}

template <typename Source, typename Result>
ffi::Future ExecuteCopy(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<Source>* source, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count) {
  try {
    auto state = MakeProgramState<CopyProgram<Source, Result>>(
        CopyInit<Result>{result, output_size, CopyCoversOutput(records, output_size)}, source, source_size);
    return ExecuteOwnedMapProgram(thread_pool, records, sizeof(*source), sizeof(*result),
        output_size, batch_count, {std::move(state), &CopyProgramEntry<Source, Result>()});
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map preparation exception"));
  }
}

}
