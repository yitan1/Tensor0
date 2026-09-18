#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include "kernels/avx2.inc"
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/construction.inc"
#include "layout/planning.inc"
#include "layout/blocking.inc"
#include "kernels/affine.inc"
#include "execute/map.inc"
#include "execute/update.inc"

void CheckValues(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected,
                 bool exact) {
  assert(actual.size() == expected.size());
  for (std::size_t element = 0; element < actual.size(); ++element) {
    if (!exact && std::isnan(scalar::Convert<scalar::F32, scalar::F16>(expected[element]))) {
      assert(std::isnan(scalar::Convert<scalar::F32, scalar::F16>(actual[element])));
    } else {
      assert(actual[element] == expected[element]);
    }
  }
}

template <typename SourceOp>
void CheckMapping(const layout::Record& record, const std::vector<uint16_t>& source,
                  const SourceOp& operation, uint64_t output_size = 0) {
  constexpr bool identity = std::is_same_v<SourceOp, expression::Identity<scalar::F16>>;
  const auto size = output_size == 0 ? layout::ElementCount(record) + 7 : output_size;
  std::vector<uint16_t> result(size, 0xdead), expected = result;
  std::vector<bool> selected(size);
  for (uint64_t row = 0; row < record.shape[0]; ++row) {
    for (uint64_t column = 0; column < record.shape[1]; ++column) {
      const auto source_index = record.source_offset + row * record.source_strides[0] +
          column * record.source_strides[1];
      const auto destination_index = record.destination_offset + row * record.destination_strides[0] +
          column * record.destination_strides[1];
      expected[destination_index] = operation(source[source_index]);
      selected[destination_index] = true;
    }
  }
  kernels::ExecuteMapRecord(record, source.data(), result.data(), operation);
  CheckValues(result, expected, identity);
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if constexpr (identity || std::is_same_v<SourceOp, expression::Scale<scalar::F16>>) {
    if (tensor0::stride::simd::CpuSupportsAvx2F16c()) {
      std::fill(result.begin(), result.end(), 0xdead);
      const std::size_t fast_axis = record.source_strides[0] == 1 ? 0 : 1;
      kernels::ExecuteRank2F16Transpose(
          source.data() + record.source_offset, result.data() + record.destination_offset,
          record.shape[fast_axis], record.shape[1 - fast_axis],
          record.source_strides[1 - fast_axis], record.destination_strides[fast_axis], operation);
      CheckValues(result, expected, identity);
    }
  }
#endif
  std::vector<uint16_t> base(size, 0xdead);
  if constexpr (identity) {
    ExecuteCopy<scalar::F16>({record}, source.data(), result.data(), size);
    for (std::size_t element = 0; element < size; ++element) {
      if (!selected[element]) expected[element] = 0;
    }
    CheckValues(result, expected, true);
  } else if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::F16>>) {
    ExecuteUpdate<scalar::F16, scalar::F16, scalar::F16>(
        {record}, source.data(), base.data(), result.data(), size, operation.factor, 0,
        expression::Identity<scalar::F16>{}, expression::Identity<scalar::F16>{});
    if (scalar::IsZero<scalar::F16>(operation.factor)) {
      for (std::size_t element = 0; element < size; ++element) {
        if (selected[element]) expected[element] = 0;
      }
    }
    CheckValues(result, expected, false);
    assert(std::all_of(base.begin(), base.end(), [](auto value) { return value == 0xdead; }));
  }
}

void CheckShape(uint64_t rows, uint64_t columns, bool reverse) {
  const auto size = rows * columns;
  std::vector<uint16_t> source(size + 2);
  for (std::size_t element = 0; element < source.size(); ++element) source[element] = element - 1;
  const auto original = source;
  const std::vector<int64_t> source_strides = reverse
      ? std::vector<int64_t>{static_cast<int64_t>(columns), 1}
      : std::vector<int64_t>{1, static_cast<int64_t>(rows)};
  const std::vector<int64_t> destination_strides = reverse
      ? std::vector<int64_t>{1, static_cast<int64_t>(rows)}
      : std::vector<int64_t>{static_cast<int64_t>(columns), 1};
  const auto record = layout::BuildLayout({rows, columns}, source_strides, 1,
      destination_strides, 3, source.size(), size + 7, 0);
  CheckMapping(record, source, expression::Identity<scalar::F16>{});
  for (uint16_t factor : {0x0000, 0x3c00, 0x3555, 0xbe00, 0x0001, 0x7bff, 0x7c00, 0x7e01}) {
    CheckMapping(record, source, expression::Scale<scalar::F16>{factor, {}});
  }
  CheckMapping(record, source,
      expression::Cast<scalar::F16,
          expression::Scale<scalar::F32, expression::Identity<scalar::F16>>>{{1.0007F, {}}});
  if (rows == 256 && columns == 256) {
    const auto program = layout::CompileGeneratedRecord(record, true, true, 2, 2);
    std::size_t blocks = 0;
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ++blocks;
      CheckMapping(block, source, expression::Identity<scalar::F16>{}, size + 7);
      CheckMapping(block, source, expression::Scale<scalar::F16>{0x3555, {}}, size + 7);
    });
    assert(blocks > 1);
  }
  assert(source == original);
}

int main() {
  for (uint64_t rows : {0, 1, 7, 8, 9, 15, 16, 17, 31, 33}) {
    for (uint64_t columns : {0, 1, 7, 8, 9, 15, 16, 17, 31, 33}) {
      for (bool reverse : {false, true}) CheckShape(rows, columns, reverse);
    }
  }
  for (bool reverse : {false, true}) {
    CheckShape(129, 131, reverse);
    CheckShape(256, 256, reverse);
    std::vector<uint16_t> source(600);
    std::iota(source.begin(), source.end(), uint16_t{0x3000});
    const auto record = layout::BuildLayout(
        reverse ? std::vector<uint64_t>{19, 17} : std::vector<uint64_t>{17, 19},
        reverse ? std::vector<int64_t>{29, 1} : std::vector<int64_t>{1, 29}, 1,
        reverse ? std::vector<int64_t>{1, 31} : std::vector<int64_t>{31, 1}, 3,
        source.size(), 600, 0);
    CheckMapping(record, source, expression::Identity<scalar::F16>{}, 600);
    CheckMapping(record, source, expression::Scale<scalar::F16>{0x3555, {}}, 600);
  }
}
