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

void CheckComponent(float actual, float expected) {
  if (std::isnan(expected)) {
    assert(std::isnan(actual));
  } else if (std::isinf(expected)) {
    assert(actual == expected);
  } else {
    assert(std::isfinite(actual));
    assert(std::abs(actual - expected) <= 4 * std::numeric_limits<float>::epsilon() *
        std::max(std::abs(expected), std::numeric_limits<float>::min()));
  }
}

void CheckValue(std::complex<float> actual, std::complex<float> expected) {
  CheckComponent(actual.real(), expected.real());
  CheckComponent(actual.imag(), expected.imag());
}

int main() {
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> components{0, -0.0F, 1, -1, 1.0001F, 1e-20F, 1e20F,
      std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(), infinity, -infinity, nan};
  std::vector<std::complex<float>> source;
  for (float real : components) {
    for (float imaginary : components) source.emplace_back(real, imaginary);
  }
  const std::complex<float> sentinel{-7, 3};
  const std::vector<std::complex<float>> factors{{0, 0}, {1, 0}, {-1, 0}, {0, 1},
      {1.25F, -0.75F}, {1e-20F, 1e20F}, {infinity, 0}, {0, infinity}, {nan, 2}};
  for (const auto factor : factors) {
    const expression::Scale<scalar::C64> operation{factor, {}};
    for (uint64_t count : {0, 1, 3, 4, 5, 7, 8, 9, 17, 143}) {
      std::vector<std::complex<float>> result(source.size() + 2, sentinel);
      const auto processed = tensor0::stride::simd::ExecuteContiguousC64Scale(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2Fma() ? count / 4 * 4 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) {
        CheckValue(result[element + 1], operation(source[element + 1]));
      }
      assert(result[processed + 1] == sentinel);
      kernels::ExecuteContiguousMap(source.data() + 1, result.data() + 1, count, operation);
      for (uint64_t element = 0; element < count; ++element) {
        CheckValue(result[element + 1], operation(source[element + 1]));
      }
      assert(result.front() == sentinel && result[count + 1] == sentinel);
    }
    for (int64_t stride : {1, -1, 0, 2}) {
      const auto record = layout::BuildLayout({17, 3}, {stride, 23}, stride < 0 ? 17 : 1,
          {1, 27}, 2, source.size(), 80, 0);
      std::vector<std::complex<float>> result(80, sentinel), expected = result;
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          expected[2 + row * 27 + element] = operation(source[record.source_offset + row * 23 + element * stride]);
        }
      }
      kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    }
    const auto record = layout::BuildLayout({source.size()}, {1}, 0, {1}, 0,
        source.size(), source.size(), 0);
    for (bool base_only : {false, true}) {
      auto result = source;
      {
        const auto programs = layout::PrepareGeneratedRecords(
            {record}, sizeof(*(base_only ? nullptr : result.data())), sizeof(scalar::Value<scalar::C64>));
        ExecuteUpdateBatch<scalar::C64, scalar::C64, scalar::C64>(
            programs, base_only ? nullptr : result.data(), result.data(), result.data(), result.size(),
            base_only ? std::complex<float>{} : factor, base_only ? factor : std::complex<float>{},
            expression::Identity<scalar::C64>{}, expression::Identity<scalar::C64>{});
      }
      for (std::size_t element = 0; element < result.size(); ++element) {
        CheckValue(result[element], scalar::IsZero<scalar::C64>(factor) ? std::complex<float>{}
            : scalar::IsOne<scalar::C64>(factor) ? source[element] : operation(source[element]));
      }
    }
  }
  std::vector<std::complex<float>> result(source.size());
  const expression::Scale<scalar::F32, expression::Identity<scalar::C64>> real_scale{2, {}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), real_scale);
  for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], real_scale(source[element]));
  const expression::Scale<scalar::C64, expression::Conjugate<scalar::C64>> conjugate{{1, 2}, {}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), conjugate);
  for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], conjugate(source[element]));
  const expression::Cast<scalar::C64,
      expression::Scale<scalar::C128,
          expression::Cast<scalar::C128, expression::Identity<scalar::C64>>>> wide{
              {{1.00000001, 2.00000003}, {}}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), wide);
  for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], wide(source[element]));
}
