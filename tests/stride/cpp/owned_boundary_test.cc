#include "execute/update.h"
namespace ffi = xla::ffi;
#include "thread_pool_test_support.h"
#include <bit>
#include <sys/mman.h>
#include <unistd.h>
namespace n = tensor0::stride;
namespace s = n::scalar;

void CheckSnapshotAndLifetime() {
  constexpr uint64_t size = 65536;
  for (uint64_t batches : {1, 4}) {
    testing::ThreadPool pool(4);
    std::vector<float> source(size * batches, 2), base(size * batches, 3), result(size * batches, -1);
    float alpha = 2, beta = 3;
    bool ready = false;
    {
      const auto record = n::layout::BuildLayout({size}, {1}, 0, {1}, 0, size, size, 0);
      auto future = n::ExecuteUpdate<s::F32, s::F32, s::F32>(pool.get(), {record},
          source.data(), base.data(), result.data(), size, size, batches, &alpha, 1, &beta, 1,
          n::expression::Identity<s::F32>{}, n::expression::Identity<s::F32>{});
      future.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); ready = true; });
    }
    assert(!ready);
    alpha = 5;
    beta = 7;
    pool.run_parallel();
    assert(ready);
    // Single-batch binding is synchronous; multi-batch coefficients load in workers.
    for (float value : result) assert(value == (batches == 1 ? 13 : 31));
  }
}

void CheckGuardedSource() {
  const auto bytes = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  auto* guarded = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(guarded != MAP_FAILED);
  testing::ThreadPool pool;
  float zero = 0, two = 2;
  std::vector<float> base(8, 3), result(8, -1);
  const auto record = n::layout::BuildLayout({1}, {1}, 0, {1}, 2, 8, 8, 0);
  assert(!testing::CompletedError(n::ExecuteUpdate<s::F32, s::F32, s::F32>(
      pool.get(), {record}, static_cast<const float*>(guarded), base.data(), result.data(),
      8, 8, 1, &zero, 1, &two, 1,
      n::expression::Identity<s::F32>{}, n::expression::Identity<s::F32>{})).failure());
  for (int i = 0; i < 8; ++i) assert(result[i] == (i == 2 ? 6 : 3));
  assert(munmap(guarded, bytes) == 0);
}

void CheckIntegerAndProjection() {
  testing::ThreadPool pool;
  const auto record = n::layout::BuildLayout({2, 2}, {1, 2}, 0, {2, 1}, 0, 4, 4, 0);
  const uint64_t source[] = {UINT64_MAX, UINT64_C(9007199254740993), UINT64_C(1) << 63, 7};
  const uint64_t base[] = {3, 5, 7, 11};
  uint64_t result[4] = {}, alpha = 3, beta = 2;
  assert(!testing::CompletedError(n::ExecuteUpdate<s::U64, s::U64, s::U64>(
      pool.get(), {record}, source, base, result, 4, 4, 1, &alpha, 1, &beta, 1,
      n::expression::Identity<s::U64>{}, n::expression::Identity<s::U64>{})).failure());
  for (int i = 0; i < 4; ++i) assert(result[i] == 3 * source[(i % 2) * 2 + i / 2] + 2 * base[i]);
  const std::complex<double> cs[] = {{1, 2}, {3, 4}, {5, -6}, {-7, 8}};
  const double cb[] = {1, 3, 5, 7};
  double cr[4] = {};
  const std::complex<double> ca{2, -3};
  const double factor = 2;
  assert(!testing::CompletedError(n::ExecuteUpdate<s::F64, s::C128, s::F64>(
      pool.get(), {record}, cs, cb, cr, 4, 4, 1, &ca, 1, &factor, 1,
      n::expression::Identity<s::C128>{}, n::expression::Identity<s::F64>{})).failure());
  for (int i = 0; i < 4; ++i) assert(cr[i] == (ca * cs[(i % 2) * 2 + i / 2]).real() + 2 * cb[i]);
}

int main() {
  CheckSnapshotAndLifetime();
  CheckGuardedSource();
  CheckIntegerAndProjection();
}
