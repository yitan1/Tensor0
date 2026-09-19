#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
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
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "layout/blocking.inc"
#include "ffi/descriptor.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "execute/scheduling.inc"
#include "execute/reduction.inc"
}
#include "thread_pool_test_support.h"

namespace native = tensor0::stride;
namespace layout = native::layout;
namespace scalar = native::scalar;
namespace expression = native::expression;

void Complete(testing::ThreadPool& pool, ffi::Future future, std::size_t tasks) {
  bool ready = false;
  future.OnReady([&](const std::optional<ffi::Error>& error) {
    assert(!error);
    ready = true;
  });
  assert(pool.tasks.size() == tasks);
  if (tasks != 0) {
    assert(!ready);
    pool.run_one();
    assert(!ready);
    pool.run_parallel();
  }
  assert(ready);
}

struct ObserveRight {
  using InputDtype = scalar::F64;
  using OutputDtype = scalar::F64;
  std::vector<uint64_t>* visited;
  double operator()(double value) const {
    visited->push_back(static_cast<uint64_t>(value));
    return 1;
  }
};

void CheckDotRightItemSize() {
  constexpr uint64_t side = 129, size = side * side;
  const auto record = layout::BuildLayout({side, side}, {1, side}, 0, {side, 1}, 0, size, size, 17);
  const auto program = layout::CompileGeneratedRecord(record, false, true, sizeof(float), sizeof(double));
  const auto wrong = layout::CompileGeneratedRecord(record, false, true, sizeof(float), sizeof(float));
  assert(program.blocks != wrong.blocks);
  std::vector<float> left(size, 1);
  std::vector<double> right(size);
  for (uint64_t index = 0; index < size; ++index) right[index] = index;
  std::vector<uint64_t> expected, visited;
  float reference = 0, result = 99;
  layout::ForEachGeneratedBlock(program, [&](const auto& block) {
    native::kernels::ExecuteDotRecord<scalar::F32>(block, left.data(), right.data(), &reference,
        expression::Identity<scalar::F32>{}, ObserveRight{&expected});
  });
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {record}, sizeof(*left.data()), sizeof(*right.data()), false);
    native::ExecuteDotBatch<scalar::F32>(
        programs, left.data(), right.data(), &result, expression::Identity<scalar::F32>{},
        ObserveRight{&visited});
  }
  assert(visited == expected && result == size && reference == size);
  visited.clear();
  testing::ThreadPool pool(1);
  Complete(pool, native::ExecuteDot<scalar::F32>(pool.get(), {record}, left.data(), right.data(),
      &result, size, size, 1, expression::Identity<scalar::F32>{}, ObserveRight{&visited}), 0);
  assert(visited == expected && result == size);
}

void CheckPartialPrograms() {
  constexpr uint64_t side = 513, size = side * side;
  const auto record = layout::BuildLayout({side, side}, {1, side}, 0, {side, 1}, 0, size, size, 11);
  const auto original = layout::CompileGeneratedRecord(record, false, true, 4, 4);
  std::vector<layout::GeneratedRecordProgram> domains{original};
  layout::SplitReductionDomains(&domains, 4, [](const auto& domain, auto axis) {
    return std::min(layout::AbsoluteStride(domain.source_strides[axis]),
                    layout::AbsoluteStride(domain.destination_strides[axis]));
  });
  assert(domains.size() == 4);
  for (const auto& domain : domains) {
    assert(domain.split_costs == original.split_costs);
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      assert(domain.blocks[axis] == std::min(original.blocks[axis], domain.record.shape[axis]));
    }
  }
  std::array<float, 5> result{99, 99, 99, 99, 99};
  std::atomic<uint64_t> domain_calls{0}, block_calls{0};
  testing::ThreadPool pool(4);
  auto future = native::ExecuteReductionPartials<scalar::F32>(pool.get(), std::move(domains), 4,
      result.data(), result.size(), 2, [&](const auto& program, float* partial) {
        assert(*partial == 0 && program.record.semantic_index == 11);
        ++domain_calls;
        layout::ForEachGeneratedBlock(program, [&](const auto& block) {
          ++block_calls;
          *partial += layout::ElementCount(block);
        });
      });
  assert((result == std::array<float, 5>{99, 99, 99, 99, 99}));
  Complete(pool, std::move(future), 4);
  assert(domain_calls == 4 && block_calls > domain_calls);
  assert((result == std::array<float, 5>{0, 0, size, 0, 0}));
}

void CheckOutputDomainsAndBinding() {
  constexpr uint64_t rows = 129, columns = 131, output = rows * columns, size = 4 * output;
  const auto first = layout::BuildLayout({rows, columns, 4}, {1, rows, output}, 0,
      {columns, 1, 0}, 2, size, output + 4, 7);
  auto second = first;
  second.semantic_index = 9;
  auto programs = layout::PrepareGeneratedRecords({first, second}, 4, 4, false);
  auto tasks = layout::BuildReductionOutputTasks(programs, 4);
  assert(tasks.size() == 4);
  std::size_t blocks = 0;
  for (const auto& task : tasks) {
    assert(task.size() == 2);
    for (std::size_t index = 0; index < task.size(); ++index) {
      assert(task[index].split_costs == programs[index].split_costs);
      assert(task[index].record.semantic_index == programs[index].record.semantic_index);
      layout::ForEachGeneratedBlock(task[index], [&](const auto&) { ++blocks; });
    }
  }
  assert(blocks > 8);
  std::vector<float> source(size, 1), result(output + 4, 99);
  for (int64_t threads : {1, 4}) {
    testing::ThreadPool pool(threads);
    auto owner = std::make_shared<int>(3);
    std::weak_ptr<int> lifetime = owner;
    std::atomic<uint64_t> bindings{0};
    auto future = native::ExecuteReduction<scalar::F32, scalar::F32>(pool.get(), {first, second},
        source.data(), result.data(), size, result.size(), 1,
        [owner, &bindings](std::size_t semantic_index, uint64_t batch, auto apply) {
          assert(*owner == 3 && batch == 0);
          assert(semantic_index == 7 || semantic_index == 9);
          ++bindings;
          apply(expression::Scale<scalar::F32>{semantic_index == 7 ? 1.0f : 2.0f});
        });
    owner.reset();
    if (threads == 4) {
      assert(!lifetime.expired());
      for (auto value : result) assert(value == 0);
    }
    Complete(pool, std::move(future), threads == 4 ? 4 : 0);
    assert(lifetime.expired() && bindings == (threads == 4 ? 8 : 2));
    for (uint64_t index = 0; index < result.size(); ++index) {
      assert(result[index] == (index >= 2 && index < output + 2 ? 12 : 0));
    }
  }
}

void CheckBatchedLifetimeAndPreparationFailure() {
  constexpr uint64_t size = 65536, batches = 7;
  const auto reduction = layout::BuildLayout({size}, {1}, 0, {0}, 2, size, 5, 9);
  const auto dot = layout::BuildLayout({size}, {1}, 0, {1}, 0, size, size, 9);
  std::vector<float> source(batches * size, 1), result(batches * 5, 99);
  testing::ThreadPool pool(4);
  Complete(pool, native::ExecuteReduction<scalar::F32, scalar::F32>(pool.get(), {reduction},
      source.data(), result.data(), size, 5, batches,
      [](std::size_t index, uint64_t batch, auto apply) {
        assert(index == 9);
        apply(expression::Scale<scalar::F32>{static_cast<float>(batch + 1)});
      }), 4);
  for (uint64_t batch = 0; batch < batches; ++batch) {
    for (uint64_t index = 0; index < 5; ++index) {
      assert(result[5 * batch + index] == (index == 2 ? size * (batch + 1) : 0));
    }
  }
  Complete(pool, native::ExecuteDot<scalar::F32>(pool.get(), {dot}, source.data(), source.data(),
      result.data(), size, size, batches, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), 4);
  for (uint64_t batch = 0; batch < batches; ++batch) assert(result[batch] == size);
  auto invalid = reduction;
  invalid.shape = {UINT64_MAX, 2};
  invalid.source_strides = invalid.destination_strides = {0, 0};
  const auto before = result;
  assert(testing::CompletedError(native::ExecuteReduction<scalar::F32, scalar::F32>(
      pool.get(), {invalid}, nullptr, result.data(), 0, 5, batches,
      [](std::size_t, uint64_t, auto apply) { apply(expression::Identity<scalar::F32>{}); })).failure());
  assert(testing::CompletedError(native::ExecuteDot<scalar::F32>(pool.get(), {invalid}, nullptr,
      nullptr, result.data(), 0, 0, batches,
      expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{})).failure());
  assert(result == before && pool.tasks.empty());
}

int main() {
  CheckDotRightItemSize();
  CheckPartialPrograms();
  CheckOutputDomainsAndBinding();
  CheckBatchedLifetimeAndPreparationFailure();
}
