#include "execute/update.h"
namespace ffi = xla::ffi;
#include "thread_pool_test_support.h"

namespace native = tensor0::stride;
namespace scalar = native::scalar;

struct ThrowingMove : native::expression::Identity<scalar::F32> {
  std::shared_ptr<int> state = std::make_shared<int>(0);
  ThrowingMove() = default;
  ThrowingMove(const ThrowingMove&) = default;
  ThrowingMove(ThrowingMove&&) { throw std::runtime_error("unexpected map move"); }
};

struct ThrowingCopy : native::expression::Identity<scalar::F32> {
  std::shared_ptr<int> copies = std::make_shared<int>(0);
  bool unknown = false;
  ThrowingCopy() = default;
  ThrowingCopy(const ThrowingCopy& other) : copies(other.copies), unknown(other.unknown) {
    // Copy into the typed entry is allowed; capture inside preparation must be contained.
    if (++*copies == 2) {
      if (unknown) throw 42;
      throw std::runtime_error("map copy failure");
    }
  }
};

void CheckEmptyAndSynchronous() {
  testing::ThreadPool pool;
  ThrowingMove map;
  assert(!testing::CompletedError(native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), {}, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, 0, nullptr, 0, map, map)).failure());
  float source = 2, base = 3, result = -7, alpha = 2, beta = 3;
  const auto record = native::layout::BuildLayout({1}, {1}, 0, {1}, 0, 1, 1, 0);
  assert(!testing::CompletedError(native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
      pool.get(), {record}, &source, &base, &result, 1, 1, 1, &alpha, 1, &beta, 1, map, map)).failure());
  assert(result == 13);
}

void CheckQueuedOwnership() {
  constexpr uint64_t size = 65536, batches = 4;
  testing::ThreadPool pool(4);
  std::vector<float> source(size * batches, 2), base(size * batches, 3), result(size * batches, -7);
  float alpha = 2, beta = 3;
  std::weak_ptr<int> lifetime;
  bool ready = false;
  {
    ThrowingMove map;
    lifetime = map.state;
    const auto record = native::layout::BuildLayout({1}, {1}, 0, {1}, 0, size, size, 0);
    auto future = native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
        pool.get(), {record}, source.data(), base.data(), result.data(), size, size, batches,
        &alpha, 1, &beta, 1, map, map);
    future.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); ready = true; });
    assert(!ready);
  }
  assert(!lifetime.expired());
  pool.run_parallel();
  assert(ready);
  for (uint64_t i = 0; i < size * batches; ++i) assert(result[i] == (i % size == 0 ? 13 : 3));
}

void CheckCopyFailure() {
  for (bool unknown : {false, true}) {
    testing::ThreadPool pool;
    ThrowingCopy map;
    map.unknown = unknown;
    const auto error = testing::CompletedError(native::ExecuteUpdate<scalar::F32, scalar::F32, scalar::F32>(
        pool.get(), {}, nullptr, nullptr, nullptr, 0, 0, 0, nullptr, 0, nullptr, 0,
        map, native::expression::Identity<scalar::F32>{}));
    assert(error.failure());
    assert(error.message() == (unknown ? "tensor0-native: unknown map preparation exception"
                                      : "tensor0-native: map copy failure"));
  }
}

int main() {
  CheckEmptyAndSynchronous();
  CheckQueuedOwnership();
  CheckCopyFailure();
}
