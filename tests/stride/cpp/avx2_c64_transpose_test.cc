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

void CheckComponent(float actual, float expected, bool exact) {
  if (exact) {
    assert(std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected));
  } else if (std::isnan(expected)) {
    assert(std::isnan(actual));
  } else if (std::isinf(expected)) {
    assert(actual == expected);
  } else {
    assert(std::isfinite(actual));
    assert(std::abs(actual - expected) <= 4 * std::numeric_limits<float>::epsilon() *
        std::max(std::abs(expected), std::numeric_limits<float>::min()));
  }
}

template <typename SourceOp>
void CheckMapping(const layout::Record& record, const std::vector<std::complex<float>>& source,
                  uint64_t output_size, const SourceOp& operation) {
  constexpr bool identity = std::is_same_v<SourceOp, expression::Identity<scalar::C64>>;
  const std::complex<float> sentinel{7, -3};
  std::vector<std::complex<float>> result(output_size, sentinel), expected = result;
  std::vector<bool> selected(output_size);
  for (uint64_t row = 0; row < record.shape[0]; ++row) {
    for (uint64_t column = 0; column < record.shape[1]; ++column) {
      const auto input = record.source_offset + row * record.source_strides[0] + column * record.source_strides[1];
      const auto output = record.destination_offset + row * record.destination_strides[0] + column * record.destination_strides[1];
      expected[output] = operation(source[input]);
      selected[output] = true;
    }
  }
  const auto check = [&] {
    for (std::size_t element = 0; element < result.size(); ++element) {
      CheckComponent(result[element].real(), expected[element].real(), identity);
      CheckComponent(result[element].imag(), expected[element].imag(), identity);
    }
  };
  kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
  check();
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if constexpr (identity || std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
    if (tensor0::stride::simd::CpuSupportsAvx2Fma()) {
      std::fill(result.begin(), result.end(), sentinel);
      const std::size_t axis = record.source_strides[0] == 1 ? 0 : 1;
      kernels::ExecuteRank2C64Transpose(source.data() + record.source_offset,
          result.data() + record.destination_offset, record.shape[axis], record.shape[1 - axis],
          record.source_strides[1 - axis], record.destination_strides[axis], operation);
      check();
    }
  }
#endif
  if constexpr (identity) {
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(scalar::Value<scalar::C64>), sizeof(scalar::Value<scalar::C64>));
      ExecuteCopyBatch<scalar::C64>(programs, source.data(), result.data(), output_size);
    }
    for (std::size_t element = 0; element < result.size(); ++element) {
      if (!selected[element]) expected[element] = {};
    }
    check();
  } else if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
    const std::vector<std::complex<float>> base(output_size, sentinel);
    {
      const auto programs = layout::PrepareGeneratedRecords(
          {record}, sizeof(*source.data()), sizeof(scalar::Value<scalar::C64>));
      ExecuteUpdateBatch<scalar::C64, scalar::C64, scalar::C64>(
          programs, source.data(), base.data(), result.data(), output_size, operation.factor, {},
          expression::Identity<scalar::C64>{}, expression::Identity<scalar::C64>{});
    }
    for (std::size_t element = 0; element < result.size(); ++element) {
      if (selected[element] && scalar::IsZero<scalar::C64>(operation.factor)) expected[element] = {};
    }
    if (scalar::IsOne<scalar::C64>(operation.factor)) {
      for (uint64_t row = 0; row < record.shape[0]; ++row) {
        for (uint64_t column = 0; column < record.shape[1]; ++column) {
          expected[record.destination_offset + row * record.destination_strides[0] + column * record.destination_strides[1]] =
              source[record.source_offset + row * record.source_strides[0] + column * record.source_strides[1]];
        }
      }
    }
    check();
  }
}

void CheckShape(uint64_t rows, uint64_t columns, int64_t padding, bool reverse) {
  const std::vector<int64_t> source_strides = reverse
      ? std::vector<int64_t>{static_cast<int64_t>(columns) + padding, 1}
      : std::vector<int64_t>{1, static_cast<int64_t>(rows) + padding};
  const std::vector<int64_t> destination_strides = reverse
      ? std::vector<int64_t>{1, static_cast<int64_t>(rows) + padding}
      : std::vector<int64_t>{static_cast<int64_t>(columns) + padding, 1};
  const auto source_size = 3 + (rows && columns ? (rows - 1) * source_strides[0] + (columns - 1) * source_strides[1] : 0);
  const auto output_size = 8 + (rows && columns ? (rows - 1) * destination_strides[0] + (columns - 1) * destination_strides[1] : 0);
  const std::vector<uint32_t> bits{0, 0x80000000U, 0x7f800000U, 0xff800000U, 0x7fc00035U,
      0x7f800035U, 0x3f800001U, 0xbf800001U, 1, 0x7f7fffffU, 0x41234567U};
  std::vector<std::complex<float>> source(source_size);
  for (std::size_t element = 0; element < source.size(); ++element) {
    source[element] = {std::bit_cast<float>(bits[element % bits.size()]),
                       std::bit_cast<float>(bits[(element / bits.size()) % bits.size()])};
  }
  const auto record = layout::BuildLayout({rows, columns}, source_strides, 1,
      destination_strides, 3, source_size, output_size, 0);
  CheckMapping(record, source, output_size, expression::Identity<scalar::C64>{});
  for (const auto factor : std::vector<std::complex<float>>{{0, 0}, {1, 0}, {1.25F, -0.75F},
      {std::numeric_limits<float>::infinity(), 0}, {std::numeric_limits<float>::quiet_NaN(), 2}}) {
    CheckMapping(record, source, output_size, expression::Scale<scalar::C64>{factor, {}});
  }
  CheckMapping(record, source, output_size, expression::Scale<scalar::F32, expression::Identity<scalar::C64>>{2, {}});
  CheckMapping(record, source, output_size, expression::Scale<scalar::C64, expression::Conjugate<scalar::C64>>{{1, 2}, {}});
  const expression::Cast<scalar::C64,
      expression::Scale<scalar::C128,
          expression::Cast<scalar::C128, expression::Identity<scalar::C64>>>> wide{
              {{1.00000001, 2.00000003}, {}}};
  CheckMapping(record, source, output_size, wide);
  if (rows == 129) {
    const auto program = layout::CompileGeneratedRecord(record, true, true, 8, 8);
    std::size_t count = 0;
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ++count;
      CheckMapping(block, source, output_size, expression::Identity<scalar::C64>{});
      CheckMapping(block, source, output_size, expression::Scale<scalar::C64>{{1.25F, -0.75F}, {}});
    });
    assert(count > 1);
  }
}

int main() {
  for (uint64_t rows : {0, 1, 3, 4, 5, 15, 16, 17, 31, 33}) {
    for (uint64_t columns : {0, 1, 3, 4, 5, 15, 16, 17, 31, 33}) {
      for (bool reverse : {false, true}) {
        for (int64_t padding : {0, 3}) CheckShape(rows, columns, padding, reverse);
      }
    }
  }
  for (bool reverse : {false, true}) CheckShape(129, 131, 3, reverse);
}
