#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"
#include "layout/traversal.h"
#include "layout/blocking.h"
#include "ffi/prepared.h"
#include "execute/scheduling.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "kernels/generic.h"
#include "kernels/specialized.h"
#include "kernels/dispatch.h"
#include "execute/reduction.h"

using namespace tensor0::stride;
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;











#include "thread_pool_test_support.h"
#include "reduction_input.inc"

template <typename Dtype>
struct Observed {
  using InputDtype = Dtype;
  using OutputDtype = Dtype;

  uint64_t* calls;

  scalar::Value<Dtype> operator()(scalar::Value<Dtype> value) const {
    ++*calls;
    return expression::Scale<Dtype>{scalar::Convert<Dtype, scalar::S32>(2)}(value);
  }
};

template <typename Dtype>
void CheckRecords(bool optimize) {
  using Value = scalar::Value<Dtype>;
  const ReductionInput first{7, 0, 2, {2, 3}, {3, 1}, {2, 1}, {3, 1}, {false, true}};
  const ReductionInput empty{8, 10, 2, {2, 0}, {1, 1}, {2, 1}, {3, 1}, {false, true}};
  const ReductionInput second{9, 6, 5, {2, 2}, {2, 1}, {2, 1}, {-3, 1}, {false, true}};
  const std::array<Value, 10> source{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const auto original_source = source;
  std::vector<layout::Record> records{
      Build(first, 10, 8), Build(empty, 10, 8), Build(second, 10, 8)};
  if (optimize) {
    for (auto& record : records) layout::OptimizeRecordForExecution(&record);
  }
  std::array<Value, 10> storage;
  storage.fill(7);
  for (int repeat = 0; repeat < 2; ++repeat) {
    uint64_t calls = 0;
    std::vector<std::size_t> bound;
    {
      const auto programs = layout::PrepareGeneratedRecords(
          records, sizeof(scalar::Value<Dtype>), sizeof(scalar::Value<Dtype>), false);
      ExecuteReductionBatch<Dtype, Dtype>(
          programs, source.data(), storage.data() + 1, 8,
          [&](std::size_t index, auto execute) {
            bound.push_back(index);
            execute(Observed<Dtype>{&calls});
          });
    }
    assert(calls == 10);
    assert((bound == std::vector<std::size_t>{7, 9}));
    assert((storage == std::array<Value, 10>{7, 0, 0, 50, 0, 0, 60, 0, 0, 7}));
    assert(source == original_source);
  }
  uint64_t calls = 0;
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {}, sizeof(scalar::Value<Dtype>), sizeof(scalar::Value<Dtype>), false);
    ExecuteReductionBatch<Dtype, Dtype>(
        programs, nullptr, storage.data() + 1, 8,
        [&](std::size_t, auto execute) { execute(Observed<Dtype>{&calls}); });
  }
  assert(calls == 0);
  assert((storage == std::array<Value, 10>{7, 0, 0, 0, 0, 0, 0, 0, 0, 7}));
}

template <typename Dtype>
void CheckEmpty(bool optimize) {
  using Value = scalar::Value<Dtype>;
  const ReductionInput input{7, 0, 2, {0}, {1}, {1}, {INT64_MAX}, {true}};
  auto record = Build(input, 0, 3);
  if (optimize) {
    layout::OptimizeRecordForExecution(&record);
  }
  std::array<Value, 8> storage{7, 7, 7, 7, 7, 7, 7, 7};
  uint64_t calls = 0;
  const auto bind = [&](std::size_t, uint64_t, auto execute) {
    assert(false);
    execute(Observed<Dtype>{&calls});
  };
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<Dtype, Dtype>(
        pool.get(), {record}, nullptr, storage.data() + 1, 0, 3, 2, bind)).failure());
  }
  assert(calls == 0);
  assert((storage == std::array<Value, 8>{7, 0, 0, 0, 0, 0, 0, 7}));
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<Dtype, Dtype>(
        pool.get(), {record}, nullptr, nullptr, 0, 3, 0, bind)).failure());
  }

  const ReductionInput no_output{7, 0, 0, {0, 3}, {3, 1}, {0, 1}, {1, 1}, {false, true}};
  auto empty_record = Build(no_output, 0, 0);
  if (optimize) {
    layout::OptimizeRecordForExecution(&empty_record);
  }
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<Dtype, Dtype>(
        pool.get(), {empty_record}, nullptr, nullptr, 0, 0, 3, bind)).failure());
  }
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<Dtype, Dtype>(
        pool.get(), {}, nullptr, nullptr, 0, 0, 3, bind)).failure());
  }
  assert(calls == 0);
}

void CheckOrder() {
  const auto bind = [](std::size_t, uint64_t, auto execute) { execute(expression::Identity<scalar::F32>{}); };
  const ReductionInput input{7, 0, 0, {3, 3, 2}, {2, 1, 3}, {1, 1, 1},
      {1, 1, 1}, {true, true, true}};
  auto record = Build(input, 10, 1);
  layout::OptimizeRecordForExecution(&record);
  const std::array<float, 10> source{0, 0, 1e20f, -1e20f, 1, 0, 0, 0, 0, 0};
  float result = 7;
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {record}, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
    ExecuteReductionBatch<scalar::F32, scalar::F32>(
        programs, source.data(), &result, 1,
        [&](std::size_t index, auto apply) { bind(index, 0, apply); });
  }
  assert(result == 1.0f);

  const ReductionInput scalar_input{};
  std::vector<layout::Record> records;
  for (int64_t offset = 0; offset < 3; ++offset) {
    auto scalar_record = scalar_input;
    scalar_record.source_offset = offset;
    records.push_back(Build(scalar_record, 3, 1));
  }
  const std::array<float, 3> values{1e20f, -1e20f, 1};
  {
    const auto programs = layout::PrepareGeneratedRecords(
        records, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
    ExecuteReductionBatch<scalar::F32, scalar::F32>(
        programs, values.data(), &result, 1,
        [&](std::size_t index, auto apply) { bind(index, 0, apply); });
  }
  assert(result == 1.0f);
  std::reverse(records.begin(), records.end());
  {
    const auto programs = layout::PrepareGeneratedRecords(
        records, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
    ExecuteReductionBatch<scalar::F32, scalar::F32>(
        programs, values.data(), &result, 1,
        [&](std::size_t index, auto apply) { bind(index, 0, apply); });
  }
  assert(result == 0.0f);
}

void CheckRecordGroupingBoundary() {
  const ReductionInput first{0, 0, 0, {1}, {1}, {1}, {1}, {true}};
  const ReductionInput second{1, 1, 0, {2}, {1}, {1}, {1}, {true}};
  const std::array<float, 3> source{1, 1e20f, -1e20f};
  for (bool optimize : {false, true}) {
    std::vector<layout::Record> records{Build(first, 3, 1), Build(second, 3, 1)};
    if (optimize) {
      for (auto& record : records) layout::OptimizeRecordForExecution(&record);
    }
    float result = 7;
    {
      const auto programs = layout::PrepareGeneratedRecords(
          records, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
      ExecuteReductionBatch<scalar::F32, scalar::F32>(
          programs, source.data(), &result, 1,
          [](std::size_t, auto execute) { execute(expression::Identity<scalar::F32>{}); });
    }
    assert(result == 0.0f);
  }
}

void CheckIndependentRecords(bool optimize) {
  const std::vector<ReductionInput> inputs{
      {7, 0, 0, {2}, {1}, {1}, {1}, {true}},
      {2, 2, 2, {1, 2}, {2, 1}, {1, 1}, {1, 1}, {false, true}},
      {4, 4, 3, {}, {}, {}, {}, {}},
      {6, 5, 4, {0}, {1}, {1}, {INT64_MAX}, {true}}};
  std::vector<layout::Record> records;
  for (const auto& input : inputs) {
    auto record = Build(input, 5, 5);
    if (optimize) layout::OptimizeRecordForExecution(&record);
    records.push_back(std::move(record));
  }
  const std::array<float, 10> source{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::array<float, 12> result;
  result.fill(7);
  float factor = 0.5f;
  std::vector<std::size_t> bound;
  const auto bind = [&](std::size_t index, uint64_t, auto execute) {
    bound.push_back(index);
    if (index == 7) {
      execute(expression::Identity<scalar::F32>{});
    } else if (index == 2) {
      execute(expression::Scale<scalar::F32>{factor});
    } else {
      assert(index == 4);
      execute(expression::Scale<scalar::S32, expression::Identity<scalar::F32>>{2});
    }
  };
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), records, source.data(), result.data() + 1, 5, 5, 2, bind)).failure());
  }
  assert((bound == std::vector<std::size_t>{7, 2, 4, 7, 2, 4}));
  assert((result == std::array<float, 12>{7, 3, 0, 3.5f, 10, 0, 13, 0, 8.5f, 20, 0, 7}));
  factor = 1.5f;
  bound.clear();
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), records, source.data(), result.data() + 1, 5, 5, 2, bind)).failure());
  }
  assert((bound == std::vector<std::size_t>{7, 2, 4, 7, 2, 4}));
  assert((result == std::array<float, 12>{7, 3, 0, 10.5f, 10, 0, 13, 0, 25.5f, 20, 0, 7}));
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), records, nullptr, nullptr, 5, 5, 0, bind)).failure());
  }
  assert(bound.size() == 6);
}

void CheckMixedRecordWriteback(bool optimize) {
  using Input = expression::Identity<scalar::F64>;
  using Narrow = expression::Cast<scalar::F32, Input>;
  const std::array<double, 10> source{
      -1, 1 + 0x1p-24, 0, 100000001, -100000000,
      -2, 2 + 0x1p-23, 0, 100000002, -100000000};
  const auto original_source = source;
  for (bool split : {false, true}) {
    std::vector<ReductionInput> inputs{
        {7, 0, 1, {}, {}, {}, {}, {}},
        {2, 1, 1, {2}, {1}, {1}, {1}, {true}},
        {9, 3, 3, {split ? 1u : 2u}, {1}, {1}, {1}, {true}}};
    if (split) inputs.push_back({4, 4, 3, {}, {}, {}, {}, {}});
    std::vector<layout::Record> records;
    for (const auto& input : inputs) {
      auto record = Build(input, 5, 5);
      if (optimize) layout::OptimizeRecordForExecution(&record);
      records.push_back(std::move(record));
    }
    std::array<float, 12> result;
    result.fill(7);
    for (bool narrow : {false, true}) {
      for (float factor : {1.0f, 2.0f}) {
        std::vector<std::size_t> bound;
        {
          testing::ThreadPool pool(1);
          assert(!testing::CompletedError(ExecuteReduction<scalar::F64, scalar::F32>(
              pool.get(), records, source.data(), result.data() + 1, 5, 5, 2,
              [&](std::size_t index, uint64_t, auto execute) {
                bound.push_back(index);
                if (index == 7) {
                  execute(Narrow{});
                } else if (index == 2) {
                  if (narrow) execute(expression::Scale<scalar::F32, Narrow>{factor});
                  else execute(expression::Scale<scalar::F32, Input>{factor});
                } else {
                  assert(index == 9 || index == 4);
                  if (narrow) execute(Narrow{});
                  else execute(Input{});
                }
              })).failure());
        }
        const std::vector<std::size_t> expected_bound = split
            ? std::vector<std::size_t>{7, 2, 9, 4, 7, 2, 9, 4}
            : std::vector<std::size_t>{7, 2, 9, 7, 2, 9};
        assert(bound == expected_bound);
        const float mixed = narrow ? factor - 1
            : (factor == 1 ? 0x1p-24f : 1 + 0x1p-23f);
        const float tail = narrow || split ? 0 : 1;
        assert((result == std::array<float, 12>{
            7, 0, mixed, 0, tail, 0, 0, 2 * mixed, 0, 2 * tail, 0, 7}));
        assert(source == original_source);
      }
    }
  }
}

void CheckBatchOverflow() {
  const std::array<std::array<uint64_t, 3>, 4> sizes{{
      {UINT64_MAX, 1, 2}, {0, UINT64_MAX, 2},
      {UINT64_C(1) << 62, 1, 1}, {0, UINT64_C(1) << 62, 1}}};
  for (const auto& size : sizes) {
    float result = 7;
    testing::ThreadPool pool(1);
    const auto error = testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), {}, nullptr, &result, size[0], size[1], size[2],
        [](std::size_t, uint64_t, auto execute) {
          assert(false);
          execute(expression::Identity<scalar::F32>{});
        }));
    assert(error.failure());
    assert(error.message() == "tensor0-native: reduction batch storage exceeds address range");
    assert(result == 7);
  }
}

int main() {
  for (bool optimize : {false, true}) {
    CheckRecords<scalar::S32>(optimize);
    CheckRecords<scalar::F32>(optimize);
    CheckEmpty<scalar::S32>(optimize);
    CheckEmpty<scalar::F32>(optimize);
    CheckIndependentRecords(optimize);
    CheckMixedRecordWriteback(optimize);
  }
  CheckOrder();
  CheckRecordGroupingBoundary();
  CheckBatchOverflow();
}
