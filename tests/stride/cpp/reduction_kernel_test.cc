#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/construction.inc"
#include "layout/planning.inc"
#include "kernels/affine.inc"
#include "kernels/reduction.inc"
#include "reduction_input.inc"

template <typename Dtype>
struct CountedScale {
  using InputDtype = Dtype;
  using OutputDtype = Dtype;

  uint64_t* calls;

  scalar::Value<Dtype> operator()(scalar::Value<Dtype> value) const {
    ++*calls;
    return expression::Scale<Dtype>{scalar::Convert<Dtype, scalar::S32>(2)}(value);
  }
};

template <typename Dtype>
void Check(const ReductionInput& input, uint64_t source_size, uint64_t output_size,
           const std::vector<std::size_t>& selected) {
  using Value = scalar::Value<Dtype>;
  std::vector<Value> source(source_size);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<Value>(index % 7 + 1);
  }
  const auto original_source = source;
  const auto record = Build(input, source_size, output_size);
  std::vector<Value> expected(output_size + 2, static_cast<Value>(7));
  for (auto address : selected) expected[address + 1] = 0;
  const auto count = layout::ElementCount(record);
  for (uint64_t element = 0; element < count; ++element) {
    auto remaining = element;
    int64_t source_index = input.source_offset;
    int64_t destination_index = input.destination_offset;
    for (std::size_t axis = 0; axis < input.shape.size(); ++axis) {
      const auto coordinate = static_cast<int64_t>(remaining % input.shape[axis]);
      remaining /= input.shape[axis];
      source_index += coordinate * input.source_strides[axis];
      if (!input.reduction_axes[axis]) {
        destination_index += coordinate * input.destination_strides[axis];
      }
    }
    expected[destination_index + 1] += static_cast<Value>(2) * source[source_index];
  }
  for (bool optimize : {false, true}) {
    auto traversal = record;
    auto output = OutputView(input, output_size);
    if (optimize) {
      layout::OptimizeRecordForExecution(&traversal);
      layout::OptimizeRecordForExecution(&output);
    }
    std::vector<Value> actual(output_size + 2, static_cast<Value>(7));
    kernels::ExecuteFillRecord<Dtype>(output, actual.data() + 1, {});
    uint64_t calls = 0;
    kernels::ExecuteReductionRecord<Dtype>(traversal,
        source.empty() ? nullptr : source.data(), actual.data() + 1,
        CountedScale<Dtype>{&calls});
    assert(calls == count);
    assert(actual == expected);
    assert(source == original_source);
    if (count == 0) {
      kernels::ExecuteReductionRecord<Dtype>(traversal, nullptr, nullptr,
          CountedScale<Dtype>{&calls});
      assert(calls == 0);
    }
  }
}

template <typename Dtype>
void CheckCases() {
  Check<Dtype>({}, 1, 1, {0});
  Check<Dtype>({7, 0, 2, {2, 3}, {3, 1}, {2, 1}, {3, 1}, {false, true}},
      6, 8, {2, 5});
  Check<Dtype>({7, 0, 2, {2, 0}, {1, 1}, {2, 1}, {3, 1}, {false, true}},
      0, 8, {2, 5});
  Check<Dtype>({7, 0, 2, {0}, {1}, {1}, {INT64_MAX}, {true}}, 0, 3, {2});
  Check<Dtype>({7, 0, 0, {0, 3}, {3, 1}, {0, 1}, {1, 1}, {false, true}},
      0, 0, {});
  Check<Dtype>({7, 5, 5, {2, 3}, {-3, -1}, {2, 1}, {-3, 1}, {false, true}},
      6, 8, {2, 5});
  Check<Dtype>({7, 0, 2, {2, 3}, {1, 0}, {2, 1}, {3, 1}, {false, true}},
      2, 8, {2, 5});
  Check<Dtype>({7, 0, 2, {3, 3, 2}, {2, 1, 3}, {1, 1, 1}, {1, 1, 1},
      {true, true, true}}, 10, 4, {2});
  Check<Dtype>({7, 2, 2, {1}, {INT64_MAX}, {1}, {INT64_MAX}, {false}},
      3, 3, {2});
  Check<Dtype>({7, 0, 0, std::vector<uint64_t>(70, 1), std::vector<int64_t>(70),
      std::vector<uint64_t>(70, 1), std::vector<int64_t>(70),
      std::vector<bool>(70, true)}, 1, 1, {0});

  const ReductionInput first{7, 0, 2, {2, 3}, {3, 1}, {2, 1}, {3, 1}, {false, true}};
  const ReductionInput second{8, 6, 2, {2, 2}, {2, 1}, {2, 1}, {3, 1}, {false, true}};
  const std::array<scalar::Value<Dtype>, 10> source{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  for (bool optimize : {false, true}) {
    std::array<scalar::Value<Dtype>, 8> result{7, 7, 7, 7, 7, 7, 7, 7};
    auto output = OutputView(first, result.size());
    if (optimize) layout::OptimizeRecordForExecution(&output);
    kernels::ExecuteFillRecord<Dtype>(output, result.data(), {});
    for (const auto& input : {first, second}) {
      auto record = Build(input, source.size(), result.size());
      if (optimize) layout::OptimizeRecordForExecution(&record);
      kernels::ExecuteReductionRecord<Dtype>(record, source.data(), result.data(),
          expression::Identity<Dtype>{});
    }
    assert((result == std::array<scalar::Value<Dtype>, 8>{7, 7, 21, 7, 7, 34, 7, 7}));
  }
}

int main() {
  CheckCases<scalar::S32>();
  CheckCases<scalar::F32>();
  const ReductionInput input{7, 0, 0, {3, 3, 2}, {2, 1, 3}, {1, 1, 1},
      {1, 1, 1}, {true, true, true}};
  auto record = Build(input, 10, 1);
  layout::OptimizeRecordForExecution(&record);
  const std::array<float, 10> source{0, 0, 1e20f, -1e20f, 1, 0, 0, 0, 0, 0};
  float result = 0;
  kernels::ExecuteReductionRecord<scalar::F32>(record, source.data(), &result,
      expression::Identity<scalar::F32>{});
  assert(result == 1.0f);
}
