#include <cassert>
#include <cmath>
#include <complex>
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

using Conversion = expression::Cast<scalar::F32, expression::Identity<scalar::F16>>;

void CheckValue(float actual, uint16_t input) {
  const auto expected = scalar::Convert<scalar::F32, scalar::F16>(input);
  if (std::isnan(expected)) {
    assert(std::isnan(actual));
  } else {
    assert(std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected));
  }
}

int main() {
  std::vector<uint16_t> input(65538);
  for (uint32_t bits = 0; bits < 65536; ++bits) input[bits + 1] = bits;
  std::vector<float> output(65538, -3.0F);
  kernels::ExecuteContiguousMap(input.data() + 1, output.data() + 1, 65536, Conversion{});
  for (uint32_t bits = 0; bits < 65536; ++bits) CheckValue(output[bits + 1], bits);
  assert(output.front() == -3.0F && output.back() == -3.0F);

  for (uint64_t count = 0; count < 34; ++count) {
    std::fill(output.begin(), output.end(), -3.0F);
    const auto processed = tensor0::stride::simd::ExecuteContiguousF16F32(
        input.data() + 1, output.data() + 1, count);
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    assert(processed == (tensor0::stride::simd::CpuSupportsAvx2F16c() ? count / 8 * 8 : 0));
#else
    assert(processed == 0);
#endif
    for (uint64_t element = 0; element < processed; ++element) {
      CheckValue(output[element + 1], input[element + 1]);
    }
    assert(output[processed + 1] == -3.0F);
    kernels::ExecuteContiguousMap(input.data() + 1, output.data() + 1, count, Conversion{});
    for (uint64_t element = 0; element < count; ++element) {
      CheckValue(output[element + 1], input[element + 1]);
    }
    assert(output.front() == -3.0F && output[count + 1] == -3.0F);
  }

  for (int64_t stride : {int64_t{1}, int64_t{-1}, int64_t{2}}) {
    const auto record = layout::BuildLayout(
        {17}, {stride}, stride < 0 ? 33 : 1, {1}, 2, input.size(), 21, 0);
    std::vector<float> result(21, -3.0F);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(scalar::Value<scalar::F16>), sizeof(scalar::Value<scalar::F32>));
      ExecuteCopyBatch<scalar::F16, scalar::F32>(programs, input.data(), result.data(), result.size());
    }
    for (int64_t element = 0; element < 17; ++element) {
      CheckValue(result[element + 2], input[record.source_offset + element * stride]);
    }
    assert(result.front() == 0.0F && result.back() == 0.0F);
    std::vector<float> base(21, -7.0F);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(*input.data()), sizeof(scalar::Value<scalar::F32>));
      ExecuteUpdateBatch<scalar::F32, scalar::F32, scalar::F32>(
          programs, input.data(), base.data(), result.data(), result.size(), 1.0F, 0.0F, Conversion{},
          expression::Identity<scalar::F32>{});
    }
    for (int64_t element = 0; element < 17; ++element) {
      CheckValue(result[element + 2], input[record.source_offset + element * stride]);
    }
    assert(result.front() == -7.0F && result.back() == -7.0F);
  }

  const auto rows = layout::BuildLayout({17, 3}, {1, 23}, 1, {1, 27}, 2,
                                        input.size(), 80, 0);
  std::vector<float> result(80, -3.0F);
  kernels::ExecuteMapRecord(rows, input.data(), result.data(), Conversion{});
  for (int64_t row = 0; row < 3; ++row) {
    for (int64_t element = 0; element < 17; ++element) {
      CheckValue(result[2 + row * 27 + element], input[1 + row * 23 + element]);
    }
    assert(result[1 + row * 27] == -3.0F);
    assert(result[19 + row * 27] == -3.0F);
  }
}
