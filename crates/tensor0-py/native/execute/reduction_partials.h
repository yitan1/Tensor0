#pragma once

#include "scheduling.h"
#include "../layout/blocking.h"
#include "../numeric/scalar.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor0::stride {

template <typename Accumulator>
inline constexpr bool kParallelReductionAccumulator =
    std::is_same_v<Accumulator, scalar::F32> || std::is_same_v<Accumulator, scalar::F64> ||
    std::is_same_v<Accumulator, scalar::C64> || std::is_same_v<Accumulator, scalar::C128>;

template <typename Accumulator>
ffi::Future ExecuteReductionPartials(
    ffi::ThreadPool thread_pool, std::vector<layout::GeneratedRecordProgram> domains, uint64_t workers,
    scalar::Value<Accumulator>* result, uint64_t output_size, uint64_t output_index,
    std::function<void(const layout::GeneratedRecordProgram&, scalar::Value<Accumulator>*)> execute) {
  using Value = scalar::Value<Accumulator>;
  struct alignas(64) Partial { Value value{}; };
  const auto count = domains.size();
  auto partials = std::make_shared<std::vector<Partial>>(count);
  return ExecuteTasks(thread_pool, count, workers,
      [domains = std::move(domains), partials, execute = std::move(execute)](uint64_t begin, uint64_t count) {
        for (uint64_t index = begin; index < begin + count; ++index) {
          execute(domains[index], &(*partials)[index].value);
        }
      }, [=] {
        Value sum{};
        for (const auto& partial : *partials) {
          sum = scalar::Add<Accumulator, Accumulator>(sum, partial.value);
        }
        std::fill_n(result, output_size, Value{});
        result[output_index] = sum;
      });
}

}
