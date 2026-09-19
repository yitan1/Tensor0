#include <cassert>
#include <cmath>
#include <cstring>
#include "kernels/avx2.inc"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "layout/blocking.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "execute/scheduling.inc"
#include "execute/map.inc"

void CheckValue(float actual, float expected) {
  if (std::isnan(expected)) {
    assert(std::isnan(actual));
  } else if (std::isinf(expected) || expected == 0) {
    assert(actual == expected && std::signbit(actual) == std::signbit(expected));
  } else {
    assert(std::isfinite(actual));
    assert(std::abs(actual - expected) <= 5e-7F * std::abs(expected) + 1e-30F);
  }
}

template <typename Operation>
void CheckMap(const std::vector<std::complex<float>>& source, const Operation& operation) {
  for (uint64_t count : {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1027}) {
    std::vector<float> result(count + 2, 7);
    kernels::ExecuteContiguousMap(source.data() + 1, result.data() + 1, count, operation);
    for (uint64_t element = 0; element < count; ++element) {
      CheckValue(result[element + 1], operation(source[element + 1]));
    }
    assert(result.front() == 7 && result.back() == 7);
  }
  for (int64_t source_stride : {1, -1, 0, 2}) {
    for (int64_t destination_stride : {1, -1, 2}) {
      const auto record = layout::BuildLayout({17, 3}, {source_stride, 37},
          source_stride < 0 ? 17 : 1, {destination_stride, 39},
          destination_stride < 0 ? 18 : 2, source.size(), 120, 0);
      std::vector<float> result(120, 7), expected = result;
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          expected[record.destination_offset + row * 39 + element * destination_stride] =
              operation(source[record.source_offset + row * 37 + element * source_stride]);
        }
      }
      kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    }
  }
}

int main() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> samples{0, -0.F, 1.0000001F, -1.0000001F,
      std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
      std::numeric_limits<float>::max(), 3.1415F, infinity, -infinity, nan};
  std::vector<std::complex<float>> source(1030);
  for (std::size_t element = 0; element < source.size(); ++element) {
    source[element] = {samples[element % samples.size()], samples[(element / samples.size()) % samples.size()]};
  }
  using Product = expression::Scale<scalar::C64>;
  using Projected = expression::Cast<scalar::F32, Product>;
  using Converted = expression::Cast<scalar::F32, expression::Identity<scalar::C64>>;
  using RealScale = expression::Scale<scalar::F32, Converted>;
  using RealProduct = expression::Scale<scalar::F32, expression::Identity<scalar::C64>>;
  using ProjectedRealProduct = expression::Cast<scalar::F32, RealProduct>;
  for (const auto factor : std::vector<std::complex<float>>{{0, 0}, {1, 0}, {-1, 0}, {0, 1},
      {1.25F, -0.75F}, {1e-20F, 1e20F}, {infinity, 1}, {0, infinity}, {nan, 2}}) {
    const Projected operation{{factor, {}}};
    CheckMap(source, operation);
    for (uint64_t count : {0, 7, 8, 9, 1027}) {
      std::vector<float> result(count + 2, 7);
      const auto processed = tensor0::stride::simd::ExecuteContiguousC64ProductReal(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2Fma() ? count - count % 8 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == 7 && result[processed + 1] == 7);
    }
    const auto record = layout::BuildLayout({17, 3}, {1, 37}, 1, {1, 39}, 2, source.size(), 120, 0);
    std::vector<float> result(120, 7), expected = result;
    kernels::ExecuteMapRecord<Product, scalar::F32>(record, source.data(), result.data(), Product{factor, {}});
    kernels::ExecuteMapRecord(record, source.data(), expected.data(), operation);
    for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    std::fill(result.begin(), result.end(), 7);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::F32>));
      ExecuteUpdateBatch<scalar::F32, scalar::C64, scalar::F32>(
          programs, source.data(), result.data(), result.data(), result.size(), factor, 0,
          expression::Identity<scalar::C64>{}, expression::Identity<scalar::F32>{});
    }
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t element = 0; element < 17; ++element) {
        const auto value = source[1 + row * 37 + element];
        expected[2 + row * 39 + element] = scalar::IsZero<scalar::C64>(factor) ? 0 :
            scalar::IsOne<scalar::C64>(factor) ? value.real() : operation(value);
      }
    }
    for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
  }
  for (float factor : {0.F, 1.F, -1.F, 1.25F, infinity, nan}) {
    const RealScale operation{factor, {}};
    CheckMap(source, operation);
    const ProjectedRealProduct projected{{factor, {}}};
    CheckMap(source, projected);
    const auto record = layout::BuildLayout({17, 3}, {1, 37}, 1, {1, 39}, 2, source.size(), 120, 0);
    std::vector<float> result(120, 7), expected = result;
    kernels::ExecuteMapRecord<RealProduct, scalar::F32>(
        record, source.data(), result.data(), RealProduct{factor, {}});
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t element = 0; element < 17; ++element) {
        expected[2 + row * 39 + element] = projected(source[1 + row * 37 + element]);
      }
    }
    for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    std::fill(result.begin(), result.end(), 7);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::F32>));
      ExecuteUpdateBatch<scalar::F32, scalar::F32, scalar::F32>(
          programs, source.data(), result.data(), result.data(), result.size(), factor, 0,
          expression::Identity<scalar::C64>{}, expression::Identity<scalar::F32>{});
    }
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t element = 0; element < 17; ++element) {
        const auto value = source[1 + row * 37 + element];
        expected[2 + row * 39 + element] = scalar::IsZero<scalar::F32>(factor) ? 0 :
            scalar::IsOne<scalar::F32>(factor) ? value.real() : projected(value);
      }
    }
    for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    for (uint64_t count : {0, 3, 4, 5, 1027}) {
      std::vector<float> result(count + 2, 7);
      const auto processed = tensor0::stride::simd::ExecuteContiguousC64RealScale(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2Fma() ? count - count % 4 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == 7 && result[processed + 1] == 7);
    }
  }
  CheckMap(source, Converted{});
  CheckMap(source, expression::Cast<scalar::F32,
      expression::Scale<scalar::C128, expression::Identity<scalar::C64>>>{{{1.00000001, 2.00000003}, {}}});
  CheckMap(source, expression::Cast<scalar::F32,
      expression::Scale<scalar::C64, expression::Conjugate<scalar::C64>>>{{{1.25F, -0.75F}, {}}});
  assert((Projected{{{1, 2}, {}}}({3, 4}) == -5));
  assert((RealScale{1, {}}({3, 4}) == 3));
}
