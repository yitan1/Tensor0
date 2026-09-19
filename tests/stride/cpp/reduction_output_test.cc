#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "reduction_input.inc"

template <typename Dtype>
void CheckFill(const layout::Record& output, uint64_t output_size,
               const std::vector<std::size_t>& selected) {
  const auto sentinel = scalar::Convert<Dtype, scalar::S32>(7);
  auto storage = std::make_unique<scalar::Value<Dtype>[]>(output_size + 2);
  std::fill_n(storage.get(), output_size + 2, sentinel);
  kernels::ExecuteFillRecord<Dtype>(output, storage.get() + 1, {});
  assert(storage[0] == sentinel && storage[output_size + 1] == sentinel);
  for (std::size_t index = 0; index < output_size; ++index) {
    const bool written = std::find(selected.begin(), selected.end(), index) != selected.end();
    assert(storage[index + 1] == (written ? scalar::Value<Dtype>{} : sentinel));
  }
}

using Dtypes = std::tuple<scalar::Pred, scalar::S8, scalar::S16, scalar::S32,
    scalar::S64, scalar::U8, scalar::U16, scalar::U32, scalar::U64,
    scalar::F16, scalar::BF16, scalar::F32, scalar::F64, scalar::C64, scalar::C128>;

template <std::size_t... Indices>
void CheckDtypes(const layout::Record& output, uint64_t output_size,
                 const std::vector<std::size_t>& selected,
                 std::index_sequence<Indices...>) {
  (CheckFill<std::tuple_element_t<Indices, Dtypes>>(output, output_size, selected), ...);
}

void Check(const ReductionInput& input, uint64_t source_size, uint64_t output_size,
           const std::vector<std::size_t>& selected) {
  const auto traversal = Build(input, source_size, output_size);
  auto output = OutputView(input, output_size);
  assert(traversal.shape == input.shape);
  for (std::size_t axis = 0; axis < input.shape.size(); ++axis) {
    assert(traversal.destination_strides[axis] ==
        (input.reduction_axes[axis] ? 0 : input.destination_strides[axis]));
  }
  assert(output.semantic_index == input.semantic_index);
  assert(output.source_offset == input.destination_offset);
  assert(output.destination_offset == input.destination_offset);
  assert(output.source_strides == output.destination_strides);
  assert(layout::ElementCount(output) == selected.size());
  CheckDtypes(output, output_size, selected, std::make_index_sequence<std::tuple_size_v<Dtypes>>{});
  layout::OptimizeRecordForExecution(&output);
  CheckDtypes(output, output_size, selected, std::make_index_sequence<std::tuple_size_v<Dtypes>>{});
  if (selected.empty()) kernels::ExecuteFillRecord<scalar::S32>(output, nullptr, {});
}

int main() {
  Check({}, 1, 1, {0});
  Check({7, 0, 2, {3}, {1}, {1}, {0}, {true}}, 3, 5, {2});
  Check({7, 0, 2, {0}, {INT64_MIN}, {1}, {0}, {true}}, 0, 5, {2});
  Check({7, 0, 2, {0}, {1}, {1}, {INT64_MAX}, {true}}, 0, 3, {2});
  Check({7, 0, 2, {0}, {INT64_MIN}, {0}, {0}, {false}}, 0, 5, {});
  Check({7, 0, 0, {2, 3}, {3, 1}, {2, 1}, {1, 0}, {false, true}}, 6, 2, {0, 1});
  Check({7, 0, 2, {2, 3}, {3, 1}, {2, 1}, {3, 1}, {false, true}}, 6, 100, {2, 5});
  Check({7, 0, 2, {2, 0}, {1, 1}, {2, 1}, {3, INT64_MIN}, {false, true}}, 0, 100, {2, 5});
  Check({7, INT64_MAX, 2, {2, 0}, {INT64_MIN, INT64_MAX}, {2, 1}, {2, 0}, {false, true}}, INT64_MAX, 8, {2, 4});
  Check({7, 0, 5, {2, 0}, {0, INT64_MIN}, {2, 1}, {-2, 0}, {false, true}}, 0, 8, {3, 5});
  Check({7, 0, 8, {0, 3}, {INT64_MAX, INT64_MIN}, {0, 1}, {1, 0}, {false, true}}, 0, 8, {});
  Check({7, 0, 1, {2, 3}, {3, 1}, {2, 3}, {4, 1}, {false, false}},
      6, 10, {1, 2, 3, 5, 6, 7});
  Check({7, 0, 3, {1, 1}, {INT64_MIN, INT64_MAX}, {1, 1}, {0, 0}, {true, false}}, 1, 5, {3});
  Check({7, 0, 3, {UINT64_MAX}, {0}, {1}, {0}, {true}}, 1, 5, {3});
  Check({7, 0, 3, {2, 2, 2}, {1, 0, 2}, {1, 1, 1}, {0, 0, 0}, {true, true, true}}, 4, 5, {3});
  ReductionInput high_rank{7, 0, 3, std::vector<uint64_t>(70, 1), std::vector<int64_t>(70), std::vector<uint64_t>(70, 1), std::vector<int64_t>(70), std::vector<bool>(70, true)};
  high_rank.reduction_axes[69] = false;
  high_rank.shape[69] = 2;
  high_rank.output_shape[69] = 2;
  high_rank.destination_strides[69] = 2;
  Check(high_rank, 1, 8, {3, 5});
  const ReductionInput input{7, 0, 2, {2, 0}, {1, 0}, {2, 1}, {2, 0}, {false, true}};
  auto traversal = Build(input, 0, 8);
  auto output = OutputView(input, 8);
  layout::OptimizeRecordForExecution(&traversal);
  layout::OptimizeRecordForExecution(&output);
  assert(layout::ElementCount(traversal) == 0);
  assert(layout::ElementCount(output) == 2);
  const ReductionInput first{7, 0, 2, {2, 3}, {3, 1}, {2, 1}, {3, 1}, {false, true}};
  const ReductionInput second{8, 6, 2, {2, 2}, {2, 1}, {2, 1}, {3, 1}, {false, true}};
  const auto first_record = Build(first, 10, 100);
  const auto second_record = Build(second, 10, 100);
  assert(first_record.source_offset == 0 && second_record.source_offset == 6);
  assert(first_record.semantic_index == 7 && second_record.semantic_index == 8);
  assert(first_record.destination_strides == second_record.destination_strides);
  assert(OutputView(first, 100).shape == OutputView(second, 100).shape);
  assert(OutputView(first, 100).destination_strides == OutputView(second, 100).destination_strides);
}
