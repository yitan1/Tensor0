#include "execute/copy.h"
namespace ffi = xla::ffi;
#include "thread_pool_test_support.h"
namespace n = tensor0::stride;
namespace s = n::scalar;

template <typename Source, typename Result>
void CheckCopy() {
  constexpr uint64_t size = 65536;
  for (uint64_t batches : {1, 4}) {
    testing::ThreadPool pool(4);
    std::vector<s::Value<Source>> source(size * batches);
    std::vector<s::Value<Result>> result(size * batches, s::Value<Result>(17));
    for (uint64_t i = 0; i < source.size(); ++i) {
      if constexpr (s::kIsComplex<Source>) source[i] = {double(i % 29), -double(i % 17)};
      else source[i] = UINT64_C(9007199254740993) + i;
    }
    bool ready = false;
    {
      // Temporary geometry and invocation state must not escape by reference.
      const auto record = n::layout::BuildLayout({size / 2}, {-1}, size - 1,
          {2}, 0, size, size, 0);
      auto future = n::ExecuteCopy<Source, Result>(pool.get(), {record}, source.data(),
          result.data(), size, size, batches);
      future.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); ready = true; });
    }
    if (!ready) pool.run_parallel();
    assert(ready);
    for (uint64_t batch = 0; batch < batches; ++batch) {
      for (uint64_t i = 0; i < size; ++i) {
        s::Value<Result> expected{};
        if (i % 2 == 0) {
          const auto value = source[batch * size + size - 1 - i / 2];
          if constexpr (s::kIsComplex<Source>) expected = value.real();
          else expected = value;
        }
        assert(result[batch * size + i] == expected);
      }
    }
  }
}

void CheckEmptyAndInvalid() {
  testing::ThreadPool pool(4);
  assert(!testing::CompletedError(n::ExecuteCopy<s::U64, s::U64>(
      pool.get(), {}, nullptr, nullptr, 0, 0, 4)).failure());
  std::vector<uint64_t> output(12, 17);
  assert(!testing::CompletedError(n::ExecuteCopy<s::U64, s::U64>(
      pool.get(), {}, nullptr, output.data(), 0, 4, 3)).failure());
  for (auto value : output) assert(value == 0);
  auto invalid = n::layout::BuildLayout({1}, {1}, 0, {1}, 0, 4, 4, 0);
  invalid.shape = {UINT64_MAX, 2};
  invalid.source_strides = invalid.destination_strides = {0, 0};
  std::fill(output.begin(), output.end(), 17);
  assert(testing::CompletedError(n::ExecuteCopy<s::U64, s::U64>(
      pool.get(), {invalid}, nullptr, output.data(), 4, 4, 3)).failure());
  for (auto value : output) assert(value == 17);
  assert(pool.tasks.empty());
}

int main() {
  CheckCopy<s::U64, s::U64>();
  CheckCopy<s::C128, s::F64>();
  CheckEmptyAndInvalid();
}
