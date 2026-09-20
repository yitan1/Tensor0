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
#include "execute/copy.h"
#include "execute/update.h"
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











#include "thread_pool_test_support.h"

namespace native = tensor0::stride;
namespace layout = native::layout;
namespace scalar = native::scalar;
namespace expression = native::expression;

void Complete(testing::ThreadPool& pool, ffi::Future future, bool parallel) {
  bool ready = false;
  future.OnReady([&](const std::optional<ffi::Error>& error) {
    assert(!error);
    ready = true;
  });
  if (parallel) {
    assert(pool.tasks.size() > 1 && pool.tasks.size() <= 4);
    assert(!ready);
    pool.run_one();
    assert(!ready);
    pool.run_parallel();
  }
  assert(ready && pool.tasks.empty());
}

void CheckCopyAndUpdate() {
  constexpr int64_t rows = 513, columns = 257;
  constexpr int64_t size = rows * columns, output_size = 2 * size + 4;
  for (int64_t threads : {0, 1, 4}) {
    for (bool broadcast : {false, true}) {
      auto record = layout::BuildLayout({rows, columns}, {broadcast ? 0 : -1, rows}, rows - 1,
                                       {2 * columns, 2}, 2, size, output_size, 3);
      const auto scalar_record = layout::BuildLayout({}, {}, 0, {}, 1, size, output_size, 4);
      const std::vector<layout::Record> records{record, scalar_record};
      std::vector<double> source(size);
      std::vector<float> base(output_size), copied(output_size, 99), expected_copy(output_size, 0);
      for (int64_t index = 0; index < size; ++index) source[index] = index % 17 + 0.5;
      for (int64_t index = 0; index < output_size; ++index) base[index] = index % 13;
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
          expected_copy[2 + 2 * (row * columns + column)] =
              source[rows - 1 - (broadcast ? 0 : row) + column * rows];
        }
      }
      expected_copy[1] = source[0];
      testing::ThreadPool pool(threads);
      auto copy = native::ExecuteCopy<scalar::F64, scalar::F32>(
          pool.get(), records, source.data(), copied.data(), size, output_size, 1);
      if (threads == 4) assert(copied == std::vector<float>(output_size, 0));
      Complete(pool, std::move(copy), threads == 4);
      assert(copied == expected_copy);
      for (double alpha : {0., 1., 2.}) {
        for (int32_t beta : {0, 1, 3}) {
          auto expected = base;
          for (int64_t index = 0; index < output_size; ++index) {
            if (index == 1 || (index >= 2 && index < 2 * size + 2 && index % 2 == 0)) {
              expected[index] = alpha * expected_copy[index] + beta * base[index];
            }
          }
          for (bool alias : {false, true}) {
            auto result = alias ? base : std::vector<float>(output_size, 99);
            const auto* original_base = alias ? result.data() : base.data();
            auto update = native::ExecuteUpdate<scalar::F32, scalar::F64, scalar::S32>(
                pool.get(), records, alpha == 0 ? nullptr : source.data(), original_base,
                result.data(), size, output_size, 1, &alpha, 1, &beta, 1,
                expression::Identity<scalar::F64>{}, expression::Identity<scalar::F32>{});
            if (threads == 4) assert(result == base);
            Complete(pool, std::move(update), threads == 4);
            assert(result == expected);
          }
        }
      }
    }
  }
}

void CheckAliasedStorageAndLifetime() {
  constexpr int64_t size = 200003;
  auto record = layout::BuildLayout({size}, {-2}, 2 * size, {-2}, 2 * size,
                                   2 * size + 2, 2 * size + 2, 0);
  const std::vector<layout::Record> records{record};
  std::vector<float> storage(2 * size + 2, 3);
  float alpha = 2, beta = 1;
  testing::ThreadPool pool(4);
  auto future = native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), records, storage.data(), storage.data(), storage.data(),
      storage.size(), storage.size(), 1, &alpha, 1, &beta, 1,
      expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  alpha = beta = 99;
  Complete(pool, std::move(future), true);
  for (int64_t index = 0; index < static_cast<int64_t>(storage.size()); ++index) {
    assert(storage[index] == ((index > 0 && index % 2 == 0) ? 9 : 3));
  }
  bool ready = false;
  auto owner = std::make_shared<int>(7);
  std::weak_ptr<int> lifetime = owner;
  auto pending = native::ExecuteMapTasks(pool.get(), layout::PrepareGeneratedRecords(records, 4, 4), [] {},
      [owner](const auto& block) {
        assert(*owner == 7);
        if (block.source_offset == 2 * size) throw std::runtime_error("injected failure");
      });
  owner.reset();
  pending.OnReady([&](const std::optional<ffi::Error>& error) {
    assert(error && error->failure());
    ready = true;
  });
  assert(!lifetime.expired() && !ready);
  pool.run_one();
  assert(!ready);
  pool.run_parallel();
  assert(ready && lifetime.expired());
}

void CheckEmptyAndBatchScheduling() {
  testing::ThreadPool pool(4);
  std::vector<float> result(9, 99), base(9, 3);
  const float alpha = 2, beta = 1;
  const std::vector<layout::Record> empty;
  Complete(pool, native::ExecuteCopy<scalar::F32, scalar::F32>(
      pool.get(), empty, nullptr, result.data(), 0, 9, 1), false);
  assert(result == std::vector<float>(9, 0));
  Complete(pool, native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), empty, nullptr, base.data(), result.data(), 0, 9, 1,
      &alpha, 1, &beta, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), false);
  assert(result == base);
  for (auto batches : {0, 1, 7}) {
    std::vector<int> visits(batches);
    auto future = native::ExecuteBatchTasks(pool.get(), batches, 65536,
        [&](auto begin, auto count) {
          for (auto batch = begin; batch < begin + count; ++batch) ++visits[batch];
        });
    Complete(pool, std::move(future), batches == 7);
    for (auto count : visits) assert(count == 1);
  }
}

void CheckBatchInitialization() {
  constexpr uint64_t size = 65536, batches = 7;
  const auto record = layout::BuildLayout({3}, {1}, 1, {2}, 2, size, size, 0);
  for (int64_t threads : {1, 4}) {
    for (bool empty : {false, true}) {
      const std::vector<layout::Record> records = empty
          ? std::vector<layout::Record>{} : std::vector<layout::Record>{record};
      testing::ThreadPool pool(threads);
      std::vector<float> source(batches * size), base(batches * size);
      std::vector<float> copied(batches * size, 99), expected_copy(batches * size, 0);
      std::array<float, batches> alpha{}, beta{};
      for (uint64_t batch = 0; batch < batches; ++batch) {
        alpha[batch] = batch % 3;
        beta[batch] = (batch + 1) % 3;
        for (uint64_t index = 0; index < size; ++index) {
          source[batch * size + index] = batch + index % 17;
          base[batch * size + index] = batch + index % 13;
        }
        if (!empty) {
          for (uint64_t index = 0; index < 3; ++index) {
            expected_copy[batch * size + 2 + 2 * index] = source[batch * size + 1 + index];
          }
        }
      }
      Complete(pool, native::ExecuteCopy<scalar::F32, scalar::F32>(
          pool.get(), records, source.data(), copied.data(), size, size, batches), threads == 4);
      assert(copied == expected_copy);
      auto expected = base;
      if (!empty) {
        for (uint64_t batch = 0; batch < batches; ++batch) {
          for (uint64_t index = 0; index < 3; ++index) {
            const auto destination = batch * size + 2 + 2 * index;
            expected[destination] = alpha[batch] * expected_copy[destination] +
                                    beta[batch] * base[destination];
          }
        }
      }
      for (bool alias : {false, true}) {
        auto result = alias ? base : std::vector<float>(batches * size, 99);
        Complete(pool, native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
            pool.get(), records, source.data(), alias ? result.data() : base.data(),
            result.data(), size, size, batches, alpha.data(), batches, beta.data(), batches,
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), threads == 4);
        assert(result == expected);
      }
    }
  }
}

void CheckMixedCoefficientBatches() {
  constexpr uint64_t size = 65536, batches = 7;
  const auto record = layout::BuildLayout({3}, {1}, 1, {2}, 2, size, size, 0);
  const std::array<double, batches> alpha{0, 1, 2, 3, 0, 1, 2};
  const std::array<int32_t, batches> beta{1, 0, 2, 1, 3, 0, 2};
  std::vector<float> source(size * batches, 2), base(size * batches, 3);
  for (uint64_t alpha_count : {UINT64_C(1), batches}) {
    for (uint64_t beta_count : {UINT64_C(1), batches}) {
      testing::ThreadPool pool(4);
      std::vector<float> result(size * batches, 99);
      Complete(pool, native::ExecuteUpdate<scalar::F32, scalar::F64, scalar::S32>(
          pool.get(), {record}, source.data(), base.data(), result.data(), size, size, batches,
          alpha.data(), alpha_count, beta.data(), beta_count,
          expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), true);
      for (uint64_t batch = 0; batch < batches; ++batch) {
        const float expected = 2 * alpha[alpha_count == 1 ? 0 : batch] +
                               3 * beta[beta_count == 1 ? 0 : batch];
        for (uint64_t index = 0; index < size; ++index) {
          const bool selected = index == 2 || index == 4 || index == 6;
          assert(result[batch * size + index] == (selected ? expected : 3));
        }
      }
    }
  }
}

void CheckWorkerLimit() {
  constexpr uint64_t size = UINT64_C(1) << 20;
  const auto record = layout::BuildLayout({size}, {1}, 0, {1}, 0, size, size, 0);
  testing::ThreadPool pool(4);
  for (uint64_t limit : {UINT64_C(1), UINT64_C(2), UINT64_C(8), UINT64_MAX}) {
    native::worker_limit.store(limit);
    const auto workers = std::min<uint64_t>(4, limit);
    assert(native::AvailableWorkerCount(pool.get()) == workers);
    std::atomic<uint64_t> visited{0};
    int initializations = 0;
    auto future = native::ExecuteMapTasks(pool.get(), layout::PrepareGeneratedRecords({record}, 4, 4),
        [&] { ++initializations; }, [&](const auto& domain) {
          assert(initializations == 1);
          visited.fetch_add(layout::ElementCount(domain));
        });
    assert(pool.tasks.size() == (workers == 1 ? 0 : workers));
    Complete(pool, std::move(future), workers > 1);
    assert(visited == size && initializations == 1);
    visited = 0;
    auto batches = native::ExecuteBatchTasks(pool.get(), 9, size,
        [&](auto, auto count) { visited.fetch_add(count); });
    assert(pool.tasks.size() == (workers == 1 ? 0 : workers));
    Complete(pool, std::move(batches), workers > 1);
    assert(visited == 9);
  }
  native::worker_limit.store(UINT64_MAX);
}

void CheckPreparedLifetimeAndFailure() {
  constexpr uint64_t size = 65536, batches = 7;
  testing::ThreadPool pool(4);
  std::vector<float> source(size * batches, 2), base(size * batches, 3), result(size * batches, 99);
  const float alpha = 2, beta = 0;
  const auto record = layout::BuildLayout({1}, {1}, 0, {1}, 1, size, size, 0);
  // Temporary record vectors must not be retained by asynchronous batch callbacks.
  Complete(pool, native::ExecuteCopy<scalar::F32, scalar::F32>(
      pool.get(), {record}, source.data(), result.data(), size, size, batches), true);
  for (uint64_t index = 0; index < result.size(); ++index) {
    assert(result[index] == (index % size == 1 ? 2 : 0));
  }
  Complete(pool, native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), {record}, source.data(), base.data(), result.data(), size, size, batches,
      &alpha, 1, &beta, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), true);
  for (uint64_t index = 0; index < result.size(); ++index) {
    assert(result[index] == (index % size == 1 ? 4 : 3));
  }
  auto invalid = record;
  invalid.shape = {UINT64_MAX, 2};
  invalid.source_strides = invalid.destination_strides = {0, 0};
  bool failed = false;
  auto future = native::ExecuteCopy<scalar::F32, scalar::F32>(
      pool.get(), {invalid}, nullptr, result.data(), 0, size, batches);
  future.OnReady([&](const std::optional<ffi::Error>& error) { failed = error.has_value(); });
  assert(failed && pool.tasks.empty() && result[0] == 3);
  // Preparation must fail before binding (and reading) coefficients or initializing output.
  for (uint64_t batch_count : {UINT64_C(1), batches}) {
    failed = false;
    auto update = native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
        pool.get(), {invalid}, nullptr, base.data(), result.data(), 0, size, batch_count,
        nullptr, 1, nullptr, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
    update.OnReady([&](const std::optional<ffi::Error>& error) { failed = error.has_value(); });
    assert(failed && pool.tasks.empty() && result[0] == 3);
  }
  Complete(pool, native::ExecuteCopy<scalar::F32, scalar::F32>(
      pool.get(), {}, nullptr, nullptr, 0, 0, batches), false);
  Complete(pool, native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), {}, nullptr, nullptr, nullptr, 0, 0, batches,
      &alpha, 1, &beta, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{}), false);
}

int main() {
  CheckMixedCoefficientBatches();
  CheckPreparedLifetimeAndFailure();
  CheckWorkerLimit();
  CheckBatchInitialization();
  CheckCopyAndUpdate();
  CheckAliasedStorageAndLifetime();
  CheckEmptyAndBatchScheduling();
}
