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
#include <complex>
#include <cstring>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;











template <typename Dtype>
void CheckValue(scalar::Value<Dtype> actual, scalar::Value<Dtype> expected) {
  const float actual_float = scalar::Convert<scalar::F32, Dtype>(actual);
  const float expected_float = scalar::Convert<scalar::F32, Dtype>(expected);
  if (std::isnan(expected_float)) {
    assert(std::isnan(actual_float));
  } else {
    assert(std::bit_cast<uint32_t>(actual_float) == std::bit_cast<uint32_t>(expected_float));
  }
}

template <typename Coefficient>
void CheckScale(scalar::Value<Coefficient> factor) {
  using Operation = expression::Scale<Coefficient, expression::Identity<scalar::F16>>;
  using Result = typename Operation::OutputDtype;
  const Operation operation{factor, {}};
  const auto sentinel = scalar::Convert<Result, scalar::F32>(-3.0F);
  std::vector<uint16_t> input(65541);
  for (std::size_t element = 0; element < input.size(); ++element) input[element] = element;
  std::vector<scalar::Value<Result>> output(input.size(), sentinel);
  kernels::ExecuteContiguousMap(input.data() + 1, output.data() + 1, 65539, operation);
  for (std::size_t element = 1; element < 65540; ++element) {
    CheckValue<Result>(output[element], operation(input[element]));
  }
  assert(output.front() == sentinel && output.back() == sentinel);

  for (uint64_t count = 0; count < 34; ++count) {
    std::fill(output.begin(), output.end(), sentinel);
    const auto processed = tensor0::stride::simd::ExecuteContiguousF16Scale(
        input.data() + 1, output.data() + 1, count,
        scalar::Convert<scalar::F32, Coefficient>(factor));
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    assert(processed == (tensor0::stride::simd::CpuSupportsAvx2F16c() ? count / 8 * 8 : 0));
#else
    assert(processed == 0);
#endif
    for (uint64_t element = 1; element <= processed; ++element) {
      CheckValue<Result>(output[element], operation(input[element]));
    }
    assert(output.front() == sentinel && output[processed + 1] == sentinel);
    kernels::ExecuteContiguousMap(input.data() + 1, output.data() + 1, count, operation);
    for (uint64_t element = 1; element <= count; ++element) {
      CheckValue<Result>(output[element], operation(input[element]));
    }
    assert(output[count + 1] == sentinel);
  }

  for (int64_t stride : {int64_t{1}, int64_t{-1}, int64_t{2}}) {
    const auto record = layout::BuildLayout({17}, {stride}, stride < 0 ? 33 : 1,
                                           {1}, 2, input.size(), 21, 0);
    std::vector<scalar::Value<Result>> base(21, sentinel), result(21);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(*input.data()), sizeof(scalar::Value<Result>));
      ExecuteUpdateBatch<Result, Coefficient, Coefficient>(
          programs, input.data(), base.data(), result.data(), result.size(), factor,
          scalar::Value<Coefficient>{}, expression::Identity<scalar::F16>{},
          expression::Identity<Result>{});
    }
    for (int64_t element = 0; element < 17; ++element) {
      const auto value = input[record.source_offset + element * stride];
      const auto expected = scalar::IsZero<Coefficient>(factor) ? scalar::Value<Result>{}
          : scalar::IsOne<Coefficient>(factor) ? scalar::Convert<Result, scalar::F16>(value)
          : operation(value);
      CheckValue<Result>(result[element + 2], expected);
    }
    assert(result.front() == sentinel && result.back() == sentinel);
  }

  const auto rows = layout::BuildLayout({17, 3}, {1, 23}, 1, {1, 27}, 2,
                                        input.size(), 80, 0);
  std::fill(output.begin(), output.end(), sentinel);
  kernels::ExecuteMapRecord(rows, input.data(), output.data(), operation);
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t element = 0; element < 17; ++element) {
      CheckValue<Result>(output[2 + row * 27 + element], operation(input[1 + row * 23 + element]));
    }
    assert(output[1 + row * 27] == sentinel && output[19 + row * 27] == sentinel);
  }
}

int main() {
  for (uint16_t factor : {0x0000, 0x8000, 0x3c00, 0xbc00, 0x3e00, 0x3555,
                          0x0001, 0x7bff, 0x7c00, 0xfc00, 0x7e01, 0x7c01}) {
    CheckScale<scalar::F16>(factor);
  }
  for (float factor : {0.0F, -0.0F, 1.0F, -1.0F, 1.5F, 1.234567F,
                        std::numeric_limits<float>::min(),
                        std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    CheckScale<scalar::F32>(factor);
  }

  std::vector<uint16_t> input(17, 0x3c01);
  std::vector<float> output(17);
  const expression::Cast<scalar::F32, expression::Scale<scalar::F16>> operation{{0x3c01, {}}};
  kernels::ExecuteContiguousMap(input.data(), output.data(), input.size(), operation);
  for (float value : output) {
    CheckValue<scalar::F32>(value, operation(0x3c01));
    assert((value != scalar::Multiply<scalar::F32>(
        scalar::Convert<scalar::F32, scalar::F16>(0x3c01),
        scalar::Convert<scalar::F32, scalar::F16>(0x3c01))));
  }

  const auto record = layout::BuildLayout({17}, {1}, 0, {1}, 0, 17, 17, 0);
  auto expected = input;
  for (auto& value : expected) value = scalar::Multiply<scalar::F16>(0x3555, value);
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {record}, sizeof(*input.data()), sizeof(scalar::Value<scalar::F16>));
    ExecuteUpdateBatch<scalar::F16, scalar::F16, scalar::F16>(
        programs, input.data(), input.data(), input.data(), input.size(), 0x3555, 0,
        expression::Identity<scalar::F16>{}, expression::Identity<scalar::F16>{});
  }
  assert(input == expected);
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {record}, sizeof(scalar::Value<scalar::F16>), sizeof(scalar::Value<scalar::F16>));
    ExecuteUpdateBatch<scalar::F16, scalar::F16, scalar::F16>(
        programs, nullptr, input.data(), input.data(), input.size(), 0, 0x3e00,
        expression::Identity<scalar::F16>{}, expression::Identity<scalar::F16>{});
  }
  for (std::size_t element = 0; element < input.size(); ++element) {
    CheckValue<scalar::F16>(input[element], scalar::Multiply<scalar::F16>(0x3e00, expected[element]));
  }
}
