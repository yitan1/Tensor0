#include "layout/record.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "execute/update.h"
#include "ffi/dtype.h"
#include <array>
#include <cassert>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#include "thread_pool_test_support.h"
#include "update_test_samples.h"

namespace native = tensor0::stride;
namespace layout = native::layout;
namespace scalar = native::scalar;
namespace expression = native::expression;

using update_test::Samples;
using update_test::CheckSame;

void CheckReaders() {
  constexpr std::array<ffi::DataType, 15> types{
      ffi::PRED, ffi::S8, ffi::S16, ffi::S32, ffi::S64, ffi::U8, ffi::U16, ffi::U32,
      ffi::U64, ffi::F16, ffi::BF16, ffi::F32, ffi::F64, ffi::C64, ffi::C128};
  // Every raw storage type exercises the reader; mapped types sample each
  // coefficient conversion family rather than repeating the binding matrix.
  constexpr std::array<ffi::DataType, 8> mapped_types{
      ffi::S64, ffi::U64, ffi::F16, ffi::BF16, ffi::F32, ffi::F64, ffi::C64, ffi::C128};
  for (auto raw_type : types) native::VisitScalarDtype(raw_type, [&](auto raw_tag) {
    using Raw = typename decltype(raw_tag)::type;
    for (auto mapped_type : mapped_types) native::VisitScalarDtype(mapped_type, [&](auto mapped_tag) {
      using Mapped = typename decltype(mapped_tag)::type;
      using Bound = native::UpdateCoefficient<Raw, Mapped>;
      const auto samples = Samples<Raw>();
      // vector<bool> cannot provide typed storage, so use actual scalar objects.
      for (auto sample : samples) {
        const scalar::Value<Raw> values[]{sample, scalar::Convert<Raw, scalar::F32>(1)};
        native::UpdateCoefficientReader reader{values, 2, native::ReadUpdateCoefficient<Raw, Bound>};
        for (uint64_t index = 0; index < 2; ++index) {
          scalar::Value<Bound> actual{};
          reader.load(reader.data, index, &actual);
          const auto expected = scalar::Convert<Bound, Raw>(values[index]);
          CheckSame<Bound>(actual, expected);
          assert(scalar::IsZero<Bound>(actual) == scalar::IsZero<Raw>(values[index]));
          assert(scalar::IsOne<Bound>(actual) == scalar::IsOne<Raw>(values[index]));
        }
      }
    });
  });
}

template <typename Source, typename Result, typename Alpha, typename Beta>
void CheckReaderExecution() {
  constexpr uint64_t size = 67;
  std::array<scalar::Value<Source>, size> source;
  std::array<scalar::Value<Result>, size> base, actual, expected;
  auto sources = Samples<Source>();
  auto bases = Samples<Result>();
  for (uint64_t i = 0; i < size; ++i) {
    source[i] = sources[i % sources.size()];
    base[i] = bases[(i + 5) % bases.size()];
  }
  for (const auto& record : std::vector<layout::Record>{
      layout::BuildLayout({size}, {1}, 0, {1}, 0, size, size, 0),
      layout::BuildLayout({size}, {-1}, size - 1, {1}, 0, size, size, 0),
      layout::BuildLayout({size}, {0}, 3, {1}, 0, size, size, 0),
      layout::BuildLayout({0}, {1}, size, {1}, size, size, size, 0)}) {
    for (auto a : Samples<Alpha>()) for (auto b : Samples<Beta>()) {
      const scalar::Value<Alpha> alpha = a;
      const scalar::Value<Beta> beta = b;
      expected = base;
      actual.fill({});
      native::ExecuteUpdateRecord<Result, Alpha, Beta>(record, source.data(), base.data(), expected.data(),
          alpha, beta, expression::Identity<Source>{}, expression::Identity<Result>{});
      testing::ThreadPool pool(1);
      auto error = testing::CompletedError(native::ExecuteUpdate<Result, Alpha, Beta>(
          pool.get(), {record}, source.data(), base.data(), actual.data(), size, size, 1,
          &alpha, 1, &beta, 1, expression::Identity<Source>{}, expression::Identity<Result>{}));
      assert(!error.failure());
      for (uint64_t i = 0; i < size; ++i) CheckSame<Result>(actual[i], expected[i]);
    }
  }
}

void ThrowStandard(const void*, uint64_t, void*) { throw std::runtime_error("reader failure"); }
void ThrowUnknown(const void*, uint64_t, void*) { throw 42; }

void CheckReaderFailure() {
  constexpr uint64_t size = 65536;
  for (auto load : {ThrowStandard, ThrowUnknown}) for (uint64_t batches : {1, 4}) {
    testing::ThreadPool pool(4);
    std::vector<float> source(size * batches, 2), base(size * batches, 3), result(size * batches, -7);
    float beta = 1;
    const auto record = layout::BuildLayout({1}, {1}, 0, {1}, 0, size, size, 0);
    auto future = native::ExecuteBoundUpdate<scalar::F32, scalar::F32, scalar::F32>(
        pool.get(), {record}, source.data(), base.data(), result.data(), size, size, batches,
        {nullptr, 1, load}, {&beta, 1, native::ReadUpdateCoefficient<scalar::F32, scalar::F32>},
        expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
    bool ready = false;
    future.OnReady([&](const std::optional<ffi::Error>& error) {
      assert(error && error->failure()); ready = true;
    });
    if (batches != 1) { assert(!ready); pool.run_parallel(); }
    assert(ready);
    // Binding fails before per-batch initialization or arithmetic writes.
    for (float value : result) assert(value == -7);
  }
}

int main() {
  CheckReaders();
  CheckReaderExecution<scalar::F32, scalar::F32, scalar::S64, scalar::BF16>();
  CheckReaderExecution<scalar::F16, scalar::F32, scalar::BF16, scalar::S16>();
  CheckReaderExecution<scalar::BF16, scalar::BF16, scalar::F16, scalar::F16>();
  CheckReaderExecution<scalar::S64, scalar::S64, scalar::U64, scalar::F32>();
  CheckReaderExecution<scalar::C64, scalar::C64, scalar::F16, scalar::F64>();
  CheckReaderExecution<scalar::C64, scalar::F32, scalar::S16, scalar::F32>();
  CheckReaderExecution<scalar::F64, scalar::C128, scalar::U64, scalar::F16>();
  CheckReaderFailure();
}
