#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"
#include "layout/traversal.h"
#include "layout/blocking.h"
#include "execute/scheduling.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "kernels/generic.h"
#include "kernels/specialized.h"
#include "kernels/dispatch.h"
#include "kernels/avx2.h"
#include "execute/update.h"

using namespace tensor0::stride;
#include <cassert>
#include <cmath>
#include <cstring>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;











void CheckValue(uint16_t actual, uint16_t expected) {
  if ((expected & 0x7c00) == 0x7c00 && (expected & 0x03ff) != 0) {
    assert((actual & 0x7c00) == 0x7c00 && (actual & 0x03ff) != 0);
  } else {
    assert(actual == expected);
  }
}

template <typename Operation>
void CheckMap(const std::vector<float>& source, const Operation& operation) {
  std::vector<uint16_t> result(source.size());
  kernels::ExecuteContiguousMap(source.data(), result.data(), source.size(), operation);
  for (std::size_t element = 0; element < source.size(); ++element) {
    CheckValue(result[element], operation(source[element]));
  }
}

int main() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> source{0, -0.F, infinity, -infinity, nan,
      std::bit_cast<float>(uint32_t{0x7f800035}),
      std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
      65504, 65519, 65520, 65521, -65520};
  for (uint16_t bits = 0; bits < 0x7bff; ++bits) {
    const float lower = scalar::Convert<scalar::F32, scalar::F16>(bits);
    const float upper = scalar::Convert<scalar::F32, scalar::F16>(bits + 1);
    const float midpoint = (lower + upper) * 0.5F;
    for (float value : {std::nextafter(midpoint, -infinity), midpoint,
                       std::nextafter(midpoint, infinity)}) {
      source.push_back(value);
      source.push_back(-value);
    }
  }
  using Product = expression::Scale<scalar::F32>;
  using Narrow = expression::Cast<scalar::F16, Product>;
  const uint16_t sentinel = 0x4200;
  for (float factor : {0.F, -0.F, 1.F, -1.F, 1.5F, 1.234567F, 1e-20F,
                       1e20F, infinity, nan}) {
    const Narrow operation{{factor, {}}};
    CheckMap(source, operation);
    for (uint64_t count : {0, 1, 7, 8, 9, 15, 16, 17, 1027}) {
      std::vector<uint16_t> result(count + 2, sentinel);
      const auto processed = tensor0::stride::simd::ExecuteContiguousF32ProductF16(
          source.data() + 1, result.data() + 1, count, factor);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      assert(processed == (tensor0::stride::simd::CpuSupportsAvx2F16c() ? count - count % 8 : 0));
#else
      assert(processed == 0);
#endif
      for (uint64_t element = 0; element < processed; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == sentinel && result[processed + 1] == sentinel);
      kernels::ExecuteContiguousMap(source.data() + 1, result.data() + 1, count, operation);
      for (uint64_t element = 0; element < count; ++element) CheckValue(result[element + 1], operation(source[element + 1]));
      assert(result.front() == sentinel && result.back() == sentinel);
    }
    for (int64_t stride : {1, -1, 0, 2}) {
      const auto record = layout::BuildLayout({17, 3}, {stride, 37}, stride < 0 ? 17 : 1,
          {1, 27}, 2, source.size(), 80, 0);
      std::vector<uint16_t> result(80, sentinel), expected = result;
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          expected[2 + row * 27 + element] = operation(source[record.source_offset + row * 37 + element * stride]);
        }
      }
      kernels::ExecuteMapRecord<Product, scalar::F16>(record, source.data(), result.data(), Product{factor, {}});
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
      std::fill(result.begin(), result.end(), sentinel);
      {
        const auto programs = layout::PrepareGeneratedRecords(
            {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::F16>));
        ExecuteUpdateBatch<scalar::F16, scalar::F32, scalar::F32>(
            programs, source.data(), result.data(), result.data(), result.size(), factor, 0,
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::F16>{});
      }
      for (int64_t row = 0; row < 3; ++row) {
        for (int64_t element = 0; element < 17; ++element) {
          const auto value = source[record.source_offset + row * 37 + element * stride];
          expected[2 + row * 27 + element] = scalar::IsZero<scalar::F32>(factor) ? 0 :
              scalar::IsOne<scalar::F32>(factor) ? scalar::Convert<scalar::F16, scalar::F32>(value) : operation(value);
        }
      }
      for (std::size_t element = 0; element < result.size(); ++element) CheckValue(result[element], expected[element]);
    }
  }
  CheckMap(source, expression::Cast<scalar::F16, expression::Identity<scalar::F32>>{});
  CheckMap(source, expression::Cast<scalar::F16,
      expression::Scale<scalar::F64, expression::Identity<scalar::F32>>>{{1.00000001, {}}});
  CheckMap(source, expression::Scale<scalar::F16,
      expression::Cast<scalar::F16, expression::Identity<scalar::F32>>>{0x3e00, {}});
}
