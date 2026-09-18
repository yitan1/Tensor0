#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace tensor0::stride {
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/construction.inc"
#include "layout/planning.inc"
#include "layout/blocking.inc"
#include "ffi/descriptor.inc"
#include "runtime/runtime.inc"
#include "kernels/affine.inc"
#include "kernels/reduction.inc"
#include "execute/reduction.inc"
#include "execute/dot.inc"
#include "execute/reduction_tasks.inc"
}
#include "thread_pool_test_support.h"

namespace native = tensor0::stride;
namespace layout = native::layout;
namespace scalar = native::scalar;
namespace expression = native::expression;

static_assert(!native::kParallelReductionAccumulator<scalar::F16>);
static_assert(!native::kParallelReductionAccumulator<scalar::BF16>);
static_assert(!native::kParallelReductionAccumulator<scalar::S32>);
static_assert(!native::kParallelReductionAccumulator<scalar::Pred>);

void Complete(testing::ThreadPool& pool, ffi::Future future, bool parallel, bool failed = false) {
  bool ready = false;
  future.OnReady([&](const std::optional<ffi::Error>& error) {
    assert(error.has_value() == failed);
    ready = true;
  });
  if (parallel) {
    assert(pool.tasks.size() > 1 && pool.tasks.size() <= 4 && !ready);
    pool.run_one();
    assert(!ready);
    pool.run_parallel();
  }
  assert(ready && pool.tasks.empty());
}

void CheckSplits() {
  auto record = layout::BuildLayout({5, 3}, {-1, 5}, 4, {0, 0}, 2, 15, 3, 19);
  std::vector<layout::Record> domains{record};
  layout::SplitReductionDomains(&domains, 3, [](const auto& domain, auto axis) {
    return std::max<uint64_t>(1, layout::AbsoluteStride(domain.source_strides[axis]));
  });
  assert(domains.size() == 3);
  for (std::size_t index = 0; index < domains.size(); ++index) {
    assert((domains[index].shape == std::vector<uint64_t>{5, 1}));
    assert(domains[index].source_offset == 4 + static_cast<int64_t>(5 * index));
    assert(domains[index].destination_offset == 2 && domains[index].semantic_index == 19);
  }
}

void CheckReduction() {
  constexpr int64_t size = 200003;
  std::vector<double> source(size);
  for (int64_t index = 0; index < size; ++index) source[index] = (index % 7 - 3) * 0.125;
  const auto record = layout::BuildLayout({size}, {-1}, size - 1, {0}, 2, size, 5, 7);
  const auto repeated = layout::BuildLayout({size}, {0}, 3, {0}, 2, size, 5, 11);
  const std::vector<layout::Record> records{record, repeated};
  for (int64_t threads : {1, 4}) {
    for (double factor : {0., 1., 2.}) {
      testing::ThreadPool pool(threads);
      std::array<float, 5> result{99, 99, 99, 99, 99}, serial{};
      auto bind = [factor](std::size_t semantic_index, uint64_t batch, auto apply) {
        assert((semantic_index == 7 || semantic_index == 11) && batch == 0);
        if (semantic_index == 11 || factor == 0) return;
        if (factor == 1) apply(expression::Identity<scalar::F64>{});
        else apply(expression::Scale<scalar::F64, expression::Identity<scalar::F64>>{factor, {}});
      };
      native::ExecuteReduction<scalar::F64, scalar::F32>(
          records, source.data(), serial.data(), size, 5, 1, bind);
      auto future = native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
          pool.get(), records, source.data(), result.data(), size, 5, 1, bind);
      if (threads == 4) for (auto value : result) assert(value == 99);
      Complete(pool, std::move(future), threads == 4);
      assert(result == serial);
    }
  }
  testing::ThreadPool pool(4);
  std::array<float, 5> result{};
  auto cast = [](std::size_t, uint64_t, auto apply) {
    apply(expression::Cast<scalar::F32, expression::Identity<scalar::F64>>{});
  };
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), {record}, source.data(), result.data(), size, 5, 1, cast), true);
  double expected = 0;
  for (auto value : source) expected += static_cast<float>(value);
  assert(result[2] == expected);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), {record}, nullptr, result.data(), size, 5, 1,
      [](std::size_t, uint64_t, auto) {}), true);
  assert((result == std::array<float, 5>{}));

  std::vector<layout::Record> overlap{
      layout::BuildLayout({2, size / 2}, {1, 2}, 0, {1, 1}, 0, size, size, 0)};
  std::vector<float> actual(size), reference(size);
  native::ExecuteReduction<scalar::F64, scalar::F32>(
      overlap, source.data(), reference.data(), size, size, 1, cast);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), overlap, source.data(), actual.data(), size, size, 1, cast), false);
  assert(actual == reference);

  std::array<int32_t, 5> integer{}, integer_reference{};
  native::ExecuteReduction<scalar::F64, scalar::S32>(records, source.data(), integer_reference.data(), size, 5, 1, cast);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::S32>(
      pool.get(), records, source.data(), integer.data(), size, 5, 1, cast), false);
  assert(integer == integer_reference);

  auto other_destination = record;
  other_destination.destination_offset = 4;
  std::array<float, 5> different_outputs{};
  native::ExecuteReduction<scalar::F64, scalar::F32>(
      {record, other_destination}, source.data(), different_outputs.data(), size, 5, 1, cast);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), {record, other_destination}, source.data(), result.data(), size, 5, 1, cast), true);
  assert(result == different_outputs);

  auto multidimensional = layout::BuildLayout({513, 257, 1}, {-1, 513, INT64_MAX}, 512,
      {0, 0, INT64_MIN}, 2, size, 5, 7);
  std::array<float, 5> multidimensional_reference{};
  native::ExecuteReduction<scalar::F64, scalar::F32>(
      {multidimensional}, source.data(), multidimensional_reference.data(), size, 5, 1, cast);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), {multidimensional}, source.data(), result.data(), size, 5, 1, cast), true);
  assert(result == multidimensional_reference);

  result.fill(99);
  Complete(pool, native::ExecuteReductionTasks<scalar::F64, scalar::F32>(
      pool.get(), {}, nullptr, result.data(), 0, 5, 1, cast), false);
  assert((result == std::array<float, 5>{}));
}

void CheckDot() {
  constexpr int64_t size = 200003;
  std::vector<std::complex<float>> left(size);
  std::vector<float> right(size);
  for (int64_t index = 0; index < size; ++index) {
    left[index] = {static_cast<float>(index % 7 - 3) * 0.125f, 0.25f};
    right[index] = static_cast<float>(index % 3 - 1) * 0.5f;
  }
  const std::vector<layout::Record> records{
      layout::BuildLayout({size}, {-1}, size - 1, {1}, 0, size, size, 0),
      layout::BuildLayout({3}, {0}, 2, {-1}, 4, size, size, 1)};
  for (int64_t threads : {1, 4}) {
    testing::ThreadPool pool(threads);
    std::complex<float> result{99, 99}, expected{};
    native::ExecuteDot<scalar::C64>(records, left.data(), right.data(), &expected, size, size, 1,
        expression::Conjugate<scalar::C64>{}, expression::Identity<scalar::F32>{});
    auto future = native::ExecuteDotTasks<scalar::C64>(pool.get(), records, left.data(), right.data(),
        &result, size, size, 1, expression::Conjugate<scalar::C64>{}, expression::Identity<scalar::F32>{});
    Complete(pool, std::move(future), threads == 4);
    assert(result == expected);
  }
  const auto multidimensional = layout::BuildLayout({513, 257, 1}, {-1, 513, INT64_MAX}, 512,
      {257, 0, INT64_MIN}, 0, size, size, 9);
  testing::ThreadPool pool(4);
  std::complex<float> result{}, expected{};
  native::ExecuteDot<scalar::C64>({multidimensional}, left.data(), right.data(), &expected,
      size, size, 1, expression::Identity<scalar::C64>{}, expression::Identity<scalar::F32>{});
  Complete(pool, native::ExecuteDotTasks<scalar::C64>(pool.get(), {multidimensional}, left.data(),
      right.data(), &result, size, size, 1,
      expression::Identity<scalar::C64>{}, expression::Identity<scalar::F32>{}), true);
  assert(result == expected);
}

void CheckCompletion() {
  constexpr uint64_t count = 4;
  std::vector<layout::Record> domains;
  for (uint64_t index = 0; index < count; ++index) {
    domains.push_back(layout::BuildLayout({}, {}, index, {}, 0, count, 1, index));
  }
  testing::ThreadPool pool(4);
  for (bool fail : {false, true}) {
    float result = 99;
    auto lifetime = std::make_shared<int>(7);
    std::weak_ptr<int> weak = lifetime;
    auto future = native::ExecuteReductionPartials<scalar::F32>(pool.get(), domains, 4, &result, 1, 0,
        [lifetime, fail](const auto& record, float* partial) {
          assert(*lifetime == 7);
          if (fail && record.semantic_index == 1) throw std::runtime_error("injected failure");
          const std::array<float, 4> contributions{16777216, 1, -16777216, 3};
          *partial = contributions[record.semantic_index];
        });
    lifetime.reset();
    assert(!weak.expired() && result == 99);
    Complete(pool, std::move(future), true, fail);
    assert(weak.expired());
    assert(result == (fail ? 99 : 3));
  }
  for (bool unknown : {false, true}) {
    auto future = native::ExecuteTasks(pool.get(), 4, 4, [](uint64_t, uint64_t) {},
        [unknown] {
          if (unknown) throw 1;
          throw std::runtime_error("injected merge failure");
        });
    Complete(pool, std::move(future), true, true);
  }
  for (uint64_t tasks : {0, 1}) {
    int finished = 0;
    Complete(pool, native::ExecuteTasks(pool.get(), tasks, 4, [](uint64_t, uint64_t) {},
        [&] { ++finished; }), false);
    assert(finished == 1);
  }
}

template <typename Dtype>
void CheckOutputTypes() {
  using Value = scalar::Value<Dtype>;
  constexpr uint64_t rows = 129, columns = 1025, size = rows * columns;
  constexpr uint64_t output_size = 2 * rows + 4;
  auto source = std::make_unique<Value[]>(size);
  for (uint64_t index = 0; index < size; ++index) {
    source[index] = scalar::Convert<Dtype, scalar::S32>(index % 3);
  }
  const auto first = layout::BuildLayout({rows, columns}, {-1, rows}, rows - 1,
                                        {-2, 0}, 2 * rows, size, output_size, 11);
  const auto second = layout::BuildLayout({columns, 1, rows}, {rows, INT64_MAX, 0}, 0,
                                         {0, INT64_MIN, -2}, 2 * rows, size, output_size, 7);
  const std::vector<layout::Record> records{first, second};
  auto expected = std::make_unique<Value[]>(output_size);
  auto result = std::make_unique<Value[]>(output_size);
  auto bind = [](std::size_t semantic_index, uint64_t batch, auto apply) {
    assert((semantic_index == 11 || semantic_index == 7) && batch == 0);
    apply(expression::Identity<Dtype>{});
  };
  native::ExecuteReduction<Dtype, Dtype>(records, source.get(), expected.get(), size, output_size, 1, bind);
  for (int64_t threads : {1, 4}) {
    testing::ThreadPool pool(threads);
    std::fill_n(result.get(), output_size, scalar::Convert<Dtype, scalar::S32>(9));
    auto future = native::ExecuteReductionTasks<Dtype, Dtype>(
        pool.get(), records, source.get(), result.get(), size, output_size, 1, bind);
    if (threads == 4) {
      for (uint64_t index = 0; index < output_size; ++index) assert(result[index] == Value{});
    }
    Complete(pool, std::move(future), threads == 4);
    for (uint64_t index = 0; index < output_size; ++index) assert(result[index] == expected[index]);
  }
}

void CheckOutputPlanning() {
  auto first = layout::BuildLayout({5, 40000}, {40000, 1}, 0, {2, 0}, 2, 200000, 32, 9);
  auto second = layout::BuildLayout({40000, 5}, {1, 0}, 0, {0, 2}, 2, 200000, 32, 3);
  auto third = layout::BuildLayout({4, 40000}, {40000, 1}, 0, {-2, 0}, 30, 200000, 32, 4);
  const auto tasks = layout::BuildReductionOutputTasks({first, third, second}, 4);
  assert(tasks.size() == 4);
  std::array<int, 32> owners;
  owners.fill(-1);
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    const auto& task = tasks[index];
    if (task.size() == 2) {
      assert(task[0].semantic_index == 9 && task[1].semantic_index == 3);
    } else {
      assert(task.size() == 1 && task[0].semantic_index == 4);
    }
    for (const auto& record : task) {
      std::size_t axis = record.destination_strides[0] == 0 ? 1 : 0;
      for (uint64_t position = 0; position < record.shape[axis]; ++position) {
        const auto address = record.destination_offset + static_cast<int64_t>(position) * record.destination_strides[axis];
        assert(owners[address] == -1 || owners[address] == static_cast<int>(index));
        owners[address] = index;
      }
    }
  }
  for (int64_t address : {2, 4, 6, 8, 10, 24, 26, 28, 30}) assert(owners[address] != -1);
  auto interleaved = first;
  interleaved.destination_offset = 3;
  assert(layout::BuildReductionOutputTasks({first, interleaved}, 4).empty());
  auto overlap = layout::BuildLayout({2, 2, 40000}, {0, 0, 1}, 0,
                                    {1, 1, 0}, 0, 40000, 4, 1);
  assert(layout::BuildReductionOutputTasks({overlap}, 4).empty());
}

void CheckOutputRecordOrder() {
  constexpr int64_t size = 40001;
  const std::array<float, 3> source{67108864, 1, -67108864};
  std::vector<layout::Record> records;
  for (int64_t index = 0; index < 3; ++index) {
    records.push_back(layout::BuildLayout({size}, {0}, index, {1}, 0, 3, size, index));
  }
  testing::ThreadPool pool(4);
  std::vector<float> result(size, 99);
  Complete(pool, native::ExecuteReductionTasks<scalar::F32, scalar::F32>(
      pool.get(), records, source.data(), result.data(), source.size(), size, 1,
      [](std::size_t, uint64_t, auto apply) { apply(expression::Identity<scalar::F32>{}); }), true);
  for (auto value : result) assert(value == 0);
}

void CheckMultipleOutputAxes() {
  constexpr uint64_t size = 200000;
  const std::vector<layout::Record> records{
      layout::BuildLayout({2, 2, 50000}, {50000, 100000, 1}, 0,
                          {2, 8, 0}, 3, size, 20, 0),
      layout::BuildLayout({50000, 2, 2}, {1, 0, 0}, 0,
                          {0, 2, 8}, 3, size, 20, 1)};
  assert(layout::BuildReductionOutputTasks(records, 4).size() == 4);
  std::vector<float> source(size);
  for (uint64_t index = 0; index < size; ++index) source[index] = index % 7;
  std::array<float, 20> result{}, expected{};
  const auto bind = [](std::size_t, uint64_t, auto apply) { apply(expression::Identity<scalar::F32>{}); };
  native::ExecuteReduction<scalar::F32, scalar::F32>(
      records, source.data(), expected.data(), size, result.size(), 1, bind);
  testing::ThreadPool pool(4);
  Complete(pool, native::ExecuteReductionTasks<scalar::F32, scalar::F32>(pool.get(), records,
      source.data(), result.data(), size, result.size(), 1, bind), true);
  assert(result == expected);
}

void CheckWorkerLimit() {
  constexpr uint64_t size = UINT64_C(1) << 18;
  const auto reduction = layout::BuildLayout({size}, {1}, 0, {0}, 0, size, 1, 0);
  const auto dot = layout::BuildLayout({size}, {1}, 0, {1}, 0, size, size, 0);
  std::vector<float> source(size, 1);
  testing::ThreadPool pool(4);
  for (uint64_t limit : {UINT64_C(1), UINT64_C(2), UINT64_C(8), UINT64_MAX}) {
    native::worker_limit.store(limit);
    const auto workers = std::min<uint64_t>(4, limit);
    assert(native::ReductionWorkerCount(pool.get(), {reduction}) == workers);
    float result = -1;
    const std::vector<layout::Record> records{reduction};
    auto future = native::ExecuteReductionTasks<scalar::F32, scalar::F32>(pool.get(), records,
        source.data(), &result, size, 1, 1,
        [](std::size_t, uint64_t, auto apply) { apply(expression::Identity<scalar::F32>{}); });
    assert(pool.tasks.size() == (workers == 1 ? 0 : workers));
    Complete(pool, std::move(future), workers > 1);
    assert(result == size);
    const std::vector<layout::Record> dot_records{dot};
    auto product = native::ExecuteDotTasks<scalar::F32>(pool.get(), dot_records,
        source.data(), source.data(), &result, size, size, 1,
        expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
    assert(pool.tasks.size() == (workers == 1 ? 0 : workers));
    Complete(pool, std::move(product), workers > 1);
    assert(result == size);
  }
  native::worker_limit.store(UINT64_MAX);
}

int main() {
  CheckWorkerLimit();
  CheckSplits();
  CheckReduction();
  CheckDot();
  CheckCompletion();
  CheckOutputPlanning();
  CheckOutputRecordOrder();
  CheckMultipleOutputAxes();
  CheckOutputTypes<scalar::Pred>();
  CheckOutputTypes<scalar::S8>();
  CheckOutputTypes<scalar::S16>();
  CheckOutputTypes<scalar::S32>();
  CheckOutputTypes<scalar::S64>();
  CheckOutputTypes<scalar::U8>();
  CheckOutputTypes<scalar::U16>();
  CheckOutputTypes<scalar::U32>();
  CheckOutputTypes<scalar::U64>();
  CheckOutputTypes<scalar::F16>();
  CheckOutputTypes<scalar::BF16>();
  CheckOutputTypes<scalar::F32>();
  CheckOutputTypes<scalar::F64>();
  CheckOutputTypes<scalar::C64>();
  CheckOutputTypes<scalar::C128>();
}
