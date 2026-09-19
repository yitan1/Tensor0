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

void CheckValue(std::complex<float> actual, std::complex<float> expected) {
  const float actual_components[]{actual.real(), actual.imag()};
  const float expected_components[]{expected.real(), expected.imag()};
  for (std::size_t component = 0; component < 2; ++component) {
    if (std::isnan(expected_components[component])) {
      assert(std::isnan(actual_components[component]));
    } else {
      assert(std::bit_cast<uint32_t>(actual_components[component]) ==
             std::bit_cast<uint32_t>(expected_components[component]));
    }
  }
}

int main() {
  const std::vector<uint32_t> bits{0, 0x80000000U, 0x3f800001U, 0xbf800001U,
      1, 0x00800000U, 0x7f7fffffU, 0x7f800000U, 0xff800000U, 0x7fc00035U, 0x7f800035U};
  std::vector<float> source(1030);
  for (std::size_t element = 0; element < source.size(); ++element) source[element] = std::bit_cast<float>(bits[element % bits.size()]);
  const std::complex<float> sentinel{7, -3};
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  for (const auto factor : std::vector<std::complex<float>>{{0, 0}, {1, 0}, {-1, 0}, {0, 1},
      {1.25F, -0.75F}, {1e-20F, 1e20F}, {infinity, 1}, {0, infinity}, {nan, 2}}) {
    const expression::Scale<scalar::C64, expression::Identity<scalar::F32>> operation{factor, {}};
    for (uint64_t count : {0, 1, 3, 7, 8, 9, 15, 16, 17, 1027}) {
      std::vector<std::complex<float>> result(source.size(), sentinel);
      const auto processed = tensor0::stride::simd::ExecuteContiguousF32C64Scale(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2Fma() ? count / 8 * 8 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result[processed + 1] == sentinel);
      kernels::ExecuteContiguousMap(source.data() + 1, result.data() + 1, count, operation);
      for (uint64_t element = 0; element < count; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == sentinel && result[count + 1] == sentinel);
    }
    for (int64_t stride : {1, -1, 0, 2}) {
      const auto record = layout::BuildLayout({17, 3}, {stride, 23}, stride < 0 ? 17 : 1,
          {1, 27}, 2, source.size(), 80, 0);
      std::vector<std::complex<float>> base(80, sentinel), result = base, expected = base;
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          expected[2 + row * 27 + element] = operation(source[record.source_offset + row * 23 + element * stride]);
        }
      }
      kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
      {
        const auto programs = layout::PrepareGeneratedRecords(
            {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::C64>));
        ExecuteUpdateBatch<scalar::C64, scalar::C64, scalar::C64>(
            programs, source.data(), base.data(), base.data(), base.size(), factor, {},
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::C64>{});
      }
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          const auto value = source[record.source_offset + row * 23 + element * stride];
          expected[2 + row * 27 + element] = scalar::IsZero<scalar::C64>(factor) ? std::complex<float>{}
              : scalar::IsOne<scalar::C64>(factor) ? scalar::Convert<scalar::C64, scalar::F32>(value) : operation(value);
        }
      }
      for (std::size_t element = 0; element < base.size(); ++element) CheckValue(base[element], expected[element]);
    }
  }
  using RealProduct = expression::Scale<scalar::F32>;
  using Embedded = expression::Cast<scalar::C64, RealProduct>;
  for (float factor : {0.F, -0.F, 1.F, -1.F, 1.25F, 1e-20F, infinity, nan}) {
    const Embedded operation{{factor, {}}};
    for (uint64_t count : {0, 1, 7, 8, 9, 15, 16, 17, 1027}) {
      std::vector<std::complex<float>> result(count + 2, sentinel);
      const auto processed = tensor0::stride::simd::ExecuteContiguousF32ProductComplex(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2Fma() ? count - count % 8 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == sentinel && result[processed + 1] == sentinel);
      kernels::ExecuteContiguousMap(source.data() + 1, result.data() + 1, count, operation);
      for (uint64_t element = 0; element < count; ++element) {
        CheckValue(result[element + 1], operation(source[element + 1]));
        assert(result[element + 1].imag() == 0 && !std::signbit(result[element + 1].imag()));
      }
      assert(result.front() == sentinel && result.back() == sentinel);
    }
    for (int64_t stride : {1, -1, 0, 2}) {
      const auto record = layout::BuildLayout({17, 3}, {stride, 37}, stride < 0 ? 17 : 1,
          {1, 27}, 2, source.size(), 80, 0);
      std::vector<std::complex<float>> result(80, sentinel), expected = result;
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          expected[2 + row * 27 + element] = operation(source[record.source_offset + row * 37 + element * stride]);
        }
      }
      kernels::ExecuteMapRecord<RealProduct, scalar::C64>(record, source.data(), result.data(), RealProduct{factor, {}});
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
      std::fill(result.begin(), result.end(), sentinel);
      {
        const auto programs = layout::PrepareGeneratedRecords(
            {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::C64>));
        ExecuteUpdateBatch<scalar::C64, scalar::F32, scalar::F32>(
            programs, source.data(), result.data(), result.data(), result.size(), factor, 0,
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::C64>{});
      }
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          const auto value = source[record.source_offset + row * 37 + element * stride];
          expected[2 + row * 27 + element] = scalar::IsZero<scalar::F32>(factor) ? std::complex<float>{} :
              scalar::IsOne<scalar::F32>(factor) ? std::complex<float>{value, 0} : operation(value);
        }
      }
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    }
  }
  std::vector<std::complex<float>> result(source.size());
  const expression::Cast<scalar::C64, expression::Identity<scalar::F32>> convert{};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), convert);
  for (std::size_t element = 0; element < source.size(); ++element) {
    CheckValue(result[element], convert(source[element]));
    assert(result[element].imag() == 0);
  }
  const expression::Scale<scalar::C64,
      expression::Cast<scalar::C64, expression::Identity<scalar::F32>>> converted{{infinity, 1}, {}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), converted);
  for (std::size_t element = 0; element < source.size(); ++element) CheckValue(result[element], converted(source[element]));
  const expression::Cast<scalar::C64,
      expression::Scale<scalar::C128, expression::Identity<scalar::F32>>> wide{{{1.00000001, 2.00000003}, {}}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), wide);
  for (std::size_t element = 0; element < source.size(); ++element) CheckValue(result[element], wide(source[element]));
  const expression::Cast<scalar::C64,
      expression::Scale<scalar::F64, expression::Identity<scalar::F32>>> real_wide{{1.00000001, {}}};
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), real_wide);
  for (std::size_t element = 0; element < source.size(); ++element) CheckValue(result[element], real_wide(source[element]));
}
