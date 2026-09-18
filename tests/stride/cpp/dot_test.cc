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
#include "execute/dot.inc"

template <typename Dtype>
void CheckLayouts(bool optimize) {
  using Value = scalar::Value<Dtype>;
  std::vector<layout::Record> records{
      layout::BuildLayout({3}, {2}, 1, {1}, 2, 11, 13, 7),
      layout::BuildLayout({3}, {-2}, 7, {-1}, 8, 11, 13, 3),
      layout::BuildLayout({2, 3}, {1, 3}, 1, {3, 1}, 1, 11, 13, 1),
      layout::BuildLayout({2, 3}, {1, 3}, 1, {0, 0}, 2, 11, 13, 9),
      layout::BuildLayout({2, 3}, {0, 0}, 2, {1, 1}, 1, 11, 13, 2),
      layout::BuildLayout({}, {}, 2, {}, 3, 11, 13, 5),
      layout::BuildLayout({1}, {INT64_MAX}, 2, {INT64_MAX}, 3, 11, 13, 4),
      layout::BuildLayout({0}, {INT64_MAX}, 11, {INT64_MIN}, 13, 11, 13, 6)};
  std::vector<uint64_t> shape(18, 1);
  std::vector<int64_t> left_strides(18, INT64_MAX), right_strides(18, INT64_MIN);
  shape[11] = 3;
  left_strides[11] = 1;
  right_strides[11] = 0;
  records.push_back(layout::BuildLayout(shape, left_strides, 1, right_strides, 2, 11, 13, 8));
  std::array<Value, 22> left;
  std::array<Value, 26> right;
  for (std::size_t index = 0; index < left.size(); ++index) {
    left[index] = scalar::Convert<Dtype, scalar::S32>(index % 3 + 1);
  }
  for (std::size_t index = 0; index < right.size(); ++index) {
    right[index] = scalar::Convert<Dtype, scalar::S32>(index % 3 + 1);
  }
  const auto original_left = left;
  const auto original_right = right;
  for (auto record : records) {
    std::array<int32_t, 2> expected{};
    for (uint64_t element = 0; element < layout::ElementCount(record); ++element) {
      auto remainder = element;
      auto left_index = record.source_offset;
      auto right_index = record.destination_offset;
      for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
        const auto coordinate = remainder % record.shape[axis];
        remainder /= record.shape[axis];
        left_index += static_cast<int64_t>(coordinate) * record.source_strides[axis];
        right_index += static_cast<int64_t>(coordinate) * record.destination_strides[axis];
      }
      for (std::size_t batch = 0; batch < 2; ++batch) {
        expected[batch] += scalar::Convert<scalar::S32, Dtype>(left[batch * 11 + left_index]) *
                           scalar::Convert<scalar::S32, Dtype>(right[batch * 13 + right_index]);
      }
    }
    if (optimize) layout::OptimizeRecordForExecution(&record);
    const auto sentinel = scalar::Convert<Dtype, scalar::S32>(7);
    std::array<Value, 4> result{sentinel, sentinel, sentinel, sentinel};
    ExecuteDot<Dtype>({record}, left.data(), right.data(), result.data() + 1,
                     11, 13, 2, expression::Identity<Dtype>{}, expression::Identity<Dtype>{});
    assert(result.front() == sentinel && result.back() == sentinel);
    for (std::size_t batch = 0; batch < 2; ++batch) {
      assert((result[batch + 1] == scalar::Convert<Dtype, scalar::S32>(expected[batch])));
    }
  }
  assert(left == original_left && right == original_right);
}

void CheckComplex() {
  using Complex = scalar::C64;
  const std::array<std::complex<float>, 2> left{{{1, 2}, {-3, 4}}};
  const std::array<std::complex<float>, 2> right{{{5, -6}, {7, 8}}};
  const auto record = layout::BuildLayout({2}, {1}, 0, {1}, 0, 2, 2, 0);
  std::complex<float> result;
  ExecuteDot<Complex>({record}, left.data(), right.data(), &result, 2, 2, 1,
      expression::Identity<Complex>{}, expression::Identity<Complex>{});
  assert(result == std::complex<float>(-36, 8));
  ExecuteDot<Complex>({record}, left.data(), right.data(), &result, 2, 2, 1,
      expression::Conjugate<Complex>{}, expression::Identity<Complex>{});
  assert(result == std::complex<float>(4, -68));
  ExecuteDot<Complex>({record}, left.data(), left.data(), &result, 2, 2, 1,
      expression::Conjugate<Complex>{}, expression::Identity<Complex>{});
  assert(result == std::complex<float>(30, 0));
  const std::complex<float> infinite{INFINITY, 2};
  const float factor = 2;
  ExecuteDot<Complex>({layout::BuildLayout({1}, {INT64_MAX}, 0, {INT64_MAX}, 0, 1, 1, 0)},
      &infinite, &factor, &result, 1, 1, 1,
      expression::Identity<Complex>{}, expression::Identity<scalar::F32>{});
  assert(std::isinf(result.real()) && result.imag() == 4);
  const float zero = 0;
  const float infinity = INFINITY;
  float real_result;
  ExecuteDot<scalar::F32>({layout::BuildLayout({}, {}, 0, {}, 0, 1, 1, 0)}, &zero, &infinity, &real_result,
      1, 1, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  assert(std::isnan(real_result));
  const std::complex<float> unit{1, 0};
  ExecuteDot<Complex>({layout::BuildLayout({}, {}, 0, {}, 0, 1, 1, 0)}, &infinite, &unit, &result,
      1, 1, 1, expression::Identity<Complex>{}, expression::Identity<Complex>{});
  assert(std::isinf(result.real()) && std::isnan(result.imag()));
}

void CheckAccumulation() {
  const std::array<float, 3> left{1, 1e20f, -1e20f};
  const std::array<float, 3> right{1, 1, 1};
  const std::vector<layout::Record> records{
      layout::BuildLayout({1}, {1}, 0, {1}, 0, 3, 3, 7),
      layout::BuildLayout({2}, {1}, 1, {1}, 1, 3, 3, 2)};
  float result = 7;
  ExecuteDot<scalar::F32>(records, left.data(), right.data(), &result, 3, 3, 1,
      expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  assert(result == 0);
  const float independent_partial = left[1] + left[2];
  assert(left[0] + independent_partial == 1);
  const std::array<double, 2> precise{-1, 1 + 0x1p-24};
  using Input = expression::Identity<scalar::F64>;
  const std::vector<layout::Record> separate{
      layout::BuildLayout({}, {}, 0, {}, 0, 2, 3, 7),
      layout::BuildLayout({}, {}, 1, {}, 1, 2, 3, 2)};
  ExecuteDot<scalar::F32>(separate, precise.data(), right.data(), &result, 2, 3, 1,
      Input{}, expression::Identity<scalar::F32>{});
  assert(result == 0x1p-24f);
  ExecuteDot<scalar::F32>(separate, precise.data(), right.data(), &result, 2, 3, 1,
      expression::Cast<scalar::F32, Input>{}, expression::Identity<scalar::F32>{});
  assert(result == 0);
  const std::array<double, 2> cancellation{100000001, -100000000};
  auto rows = layout::BuildLayout({1, 2}, {1, 1}, 0, {1, 1}, 0, 2, 3, 0);
  ExecuteDot<scalar::F32>({rows}, cancellation.data(), right.data(), &result, 2, 3, 1,
      Input{}, expression::Identity<scalar::F32>{});
  assert(result == 0);
  layout::OptimizeRecordForExecution(&rows);
  ExecuteDot<scalar::F32>({rows}, cancellation.data(), right.data(), &result, 2, 3, 1,
      Input{}, expression::Identity<scalar::F32>{});
  assert(result == 1);
  const std::array<int8_t, 2> integers{100, 100};
  const std::array<int8_t, 2> factors{2, 2};
  int32_t integer_result;
  ExecuteDot<scalar::S32>({layout::BuildLayout({2}, {1}, 0, {1}, 0, 2, 2, 0)}, integers.data(), factors.data(),
      &integer_result, 2, 2, 1, expression::Identity<scalar::S8>{}, expression::Identity<scalar::S8>{});
  assert(integer_result == -112);
  const std::array<scalar::Value<scalar::F16>, 3> half{
      scalar::Convert<scalar::F16, scalar::F32>(2048),
      scalar::Convert<scalar::F16, scalar::F32>(1),
      scalar::Convert<scalar::F16, scalar::F32>(-2048)};
  const auto half_one = scalar::Convert<scalar::F16, scalar::F32>(1);
  scalar::Value<scalar::F16> half_result;
  const auto broadcast = layout::BuildLayout({3}, {1}, 0, {0}, 0, 3, 1, 0);
  ExecuteDot<scalar::F16>({broadcast}, half.data(), &half_one, &half_result, 3, 1, 1,
      expression::Identity<scalar::F16>{}, expression::Identity<scalar::F16>{});
  assert((scalar::Convert<scalar::F32, scalar::F16>(half_result) == 0));
  ExecuteDot<scalar::F16>({broadcast}, half.data(), right.data(), &half_result, 3, 3, 1,
      expression::Identity<scalar::F16>{}, expression::Identity<scalar::F32>{});
  assert((scalar::Convert<scalar::F32, scalar::F16>(half_result) == 1));
}

struct ObserveInput {
  using InputDtype = scalar::F64;
  using OutputDtype = scalar::F64;

  const float* destination;
  std::vector<double>* inputs;
  std::vector<float>* writebacks;

  double operator()(double value) const {
    inputs->push_back(value);
    writebacks->push_back(*destination);
    return value;
  }
};

void CheckSharedWriteback() {
  const std::array<double, 3> left{1e8, 1, -1e8};
  const std::array<double, 3> right{1, 1, 1};
  for (bool optimize : {false, true}) {
    auto dot = layout::BuildLayout({1, 3}, {1, 1}, 0, {1, 1}, 0, 3, 3, 0);
    auto reduction = layout::BuildReductionLayout(
        {1, 3}, {1, 1}, 0, {1, 1}, {1, 1}, 1, {false, true}, 3, 3, 0);
    if (optimize) {
      layout::OptimizeRecordForExecution(&dot);
      layout::OptimizeRecordForExecution(&reduction);
    }
    std::array<float, 3> result{7, 0, 7};
    std::vector<double> left_inputs, right_inputs;
    std::vector<float> left_writes, right_writes;
    const ObserveInput left_op{result.data() + 1, &left_inputs, &left_writes};
    const ObserveInput right_op{result.data() + 1, &right_inputs, &right_writes};
    kernels::ExecuteDotRecord<scalar::F32>(dot, left.data(), right.data(), result.data() + 1,
                                          left_op, right_op);
    const std::vector<float> expected_writes = optimize
        ? std::vector<float>{0, 0, 0} : std::vector<float>{0, 1e8f, 1e8f};
    assert((result == std::array<float, 3>{7, optimize ? 1.0f : 0.0f, 7}));
    assert(left_inputs == std::vector<double>(left.begin(), left.end()));
    assert(right_inputs == std::vector<double>(right.begin(), right.end()));
    assert(left_writes == expected_writes && right_writes == expected_writes);
    result[1] = 0;
    left_inputs.clear();
    left_writes.clear();
    kernels::ExecuteReductionRecord<scalar::F32>(reduction, left.data(), result.data(), left_op);
    assert((result == std::array<float, 3>{7, optimize ? 1.0f : 0.0f, 7}));
    assert(left_inputs == std::vector<double>(left.begin(), left.end()));
    assert(left_writes == expected_writes);
  }
  const auto dot = layout::BuildLayout({0}, {INT64_MAX}, 0, {INT64_MIN}, 0, 0, 0, 0);
  const auto reduction = layout::BuildReductionLayout(
      {0}, {INT64_MAX}, 0, {1}, {INT64_MIN}, 0, {true}, 0, 1, 0);
  float result = 7;
  std::vector<double> inputs;
  std::vector<float> writes;
  const ObserveInput observe{&result, &inputs, &writes};
  kernels::ExecuteDotRecord<scalar::F32>(dot, nullptr, nullptr, &result, observe, observe);
  kernels::ExecuteReductionRecord<scalar::F32>(reduction, nullptr, &result, observe);
  assert(result == 7 && inputs.empty() && writes.empty());
}

void CheckBoundaries() {
  const auto identity = expression::Identity<scalar::F32>{};
  const auto empty = layout::BuildLayout({0}, {INT64_MAX}, 0, {INT64_MIN}, 0, 0, 0, 0);
  const auto repeated = layout::BuildLayout({3}, {0}, 0, {0}, 0, 1, 1, 0);
  bool invalid_map = false;
  try { layout::ValidateInjectiveView(repeated, false, {0}); }
  catch (const std::invalid_argument&) { invalid_map = true; }
  assert(invalid_map);
  std::array<float, 4> result{7, 7, 7, 7};
  ExecuteDot<scalar::F32>({empty}, nullptr, nullptr, result.data() + 1, 0, 0, 2, identity, identity);
  assert((result == std::array<float, 4>{7, 0, 0, 7}));
  ExecuteDot<scalar::F32>({}, nullptr, nullptr, result.data() + 1, 0, 0, 2, identity, identity);
  ExecuteDot<scalar::F32>({empty}, nullptr, nullptr, nullptr, 0, 0, 0, identity, identity);
  for (const auto& sizes : std::array<std::array<uint64_t, 3>, 3>{{
           {UINT64_MAX, 1, 2}, {1, UINT64_MAX, 2}, {0, 0, UINT64_MAX}}}) {
    bool rejected = false;
    try {
      ExecuteDot<scalar::F32>({}, nullptr, nullptr, result.data(), sizes[0], sizes[1], sizes[2],
          identity, identity);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && result.front() == 7);
  }
  const auto reject = [](const std::vector<uint64_t>& shape,
                         const std::vector<int64_t>& left_strides, int64_t left_offset,
                         const std::vector<int64_t>& right_strides, int64_t right_offset) {
    bool rejected = false;
    try { layout::BuildLayout(shape, left_strides, left_offset, right_strides, right_offset, 2, 2, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  };
  reject({1}, {}, 0, {1}, 0);
  reject({1}, {1}, -1, {1}, 0);
  reject({2}, {1}, 0, {2}, 0);
  reject({2}, {INT64_MAX}, 1, {0}, 0);
  reject({0}, {0}, 3, {0}, 0);
  reject({UINT64_MAX, 2}, {0, 0}, 0, {0, 0}, 0);
}

int main() {
  for (bool optimize : {false, true}) {
    CheckLayouts<scalar::Pred>(optimize);
    CheckLayouts<scalar::S8>(optimize);
    CheckLayouts<scalar::S16>(optimize);
    CheckLayouts<scalar::S32>(optimize);
    CheckLayouts<scalar::S64>(optimize);
    CheckLayouts<scalar::U8>(optimize);
    CheckLayouts<scalar::U16>(optimize);
    CheckLayouts<scalar::U32>(optimize);
    CheckLayouts<scalar::U64>(optimize);
    CheckLayouts<scalar::F16>(optimize);
    CheckLayouts<scalar::BF16>(optimize);
    CheckLayouts<scalar::F32>(optimize);
    CheckLayouts<scalar::F64>(optimize);
    CheckLayouts<scalar::C64>(optimize);
    CheckLayouts<scalar::C128>(optimize);
  }
  CheckComplex();
  CheckAccumulation();
  CheckSharedWriteback();
  CheckBoundaries();
}
