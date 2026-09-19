#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
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
#include "execute/reduction.inc"
#include "reduction_input.inc"

template <typename Accumulator, typename SourceOp, std::size_t Count>
scalar::Value<Accumulator> Reduce(
    const std::array<scalar::Value<typename SourceOp::InputDtype>, Count>& source,
    const SourceOp& source_op, bool optimize) {
  const ReductionInput input{0, 0, 1, {Count}, {1}, {1}, {1}, {true}};
  auto record = Build(input, Count, 3);
  if (optimize) {
    layout::OptimizeRecordForExecution(&record);
  }
  const auto sentinel = scalar::Convert<Accumulator, scalar::S32>(7);
  std::array<scalar::Value<Accumulator>, 5> result{sentinel, sentinel, sentinel, sentinel, sentinel};
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {record}, sizeof(scalar::Value<typename SourceOp::InputDtype>),
        sizeof(scalar::Value<Accumulator>), false);
    ExecuteReductionBatch<typename SourceOp::InputDtype, Accumulator>(
        programs, source.data(), result.data() + 1, 3,
        [&](std::size_t index, auto execute) {
          assert(index == 0);
          execute(source_op);
        });
  }
  assert(result.front() == sentinel && result.back() == sentinel);
  assert(result[1] == scalar::Value<Accumulator>{} && result[3] == scalar::Value<Accumulator>{});
  return result[2];
}

template <typename Dtype>
void CheckDtype(bool optimize) {
  using Value = scalar::Value<Dtype>;
  const std::array<Value, 3> source{
      scalar::Convert<Dtype, scalar::S32>(1),
      scalar::Convert<Dtype, scalar::S32>(2),
      scalar::Convert<Dtype, scalar::S32>(3)};
  const auto result = Reduce<Dtype>(source, expression::Identity<Dtype>{}, optimize);
  assert((result == scalar::Convert<Dtype, scalar::S32>(6)));
  if constexpr (std::is_integral_v<Value> && !scalar::kIsHalf<Dtype> &&
                !std::is_same_v<Dtype, scalar::Pred>) {
    const std::array<Value, 2> overflow{std::numeric_limits<Value>::max(), 1};
    const auto wrapped = Reduce<Dtype>(overflow, expression::Identity<Dtype>{}, optimize);
    assert(wrapped == std::numeric_limits<Value>::min());
    if constexpr (std::is_signed_v<Value>) {
      const std::array<Value, 2> underflow{std::numeric_limits<Value>::min(), -1};
      assert(Reduce<Dtype>(underflow, expression::Identity<Dtype>{}, optimize) ==
          std::numeric_limits<Value>::max());
    }
  }
}

template <typename Dtype>
void CheckLowPrecision(float boundary, bool optimize) {
  const std::array<scalar::Value<Dtype>, 3> source{
      scalar::Convert<Dtype, scalar::F32>(boundary),
      scalar::Convert<Dtype, scalar::F32>(1),
      scalar::Convert<Dtype, scalar::F32>(1)};
  const auto rounded = Reduce<Dtype>(source, expression::Identity<Dtype>{}, optimize);
  assert((scalar::Convert<scalar::F32, Dtype>(rounded) == boundary));
  const auto widened = Reduce<scalar::F32>(source,
      expression::Cast<scalar::F32, expression::Identity<Dtype>>{}, optimize);
  assert(widened == boundary + 2);

  const float infinity = std::numeric_limits<float>::infinity();
  const std::array<scalar::Value<Dtype>, 2> special{
      scalar::Convert<Dtype, scalar::F32>(infinity),
      scalar::Convert<Dtype, scalar::F32>(-infinity)};
  const auto nan = Reduce<Dtype>(special, expression::Identity<Dtype>{}, optimize);
  assert(std::isnan((scalar::Convert<scalar::F32, Dtype>(nan))));
}

void CheckConversionOrder(bool optimize) {
  const std::array<float, 2> source{0.75f, 0.75f};
  using InputCast = expression::Cast<scalar::S32, expression::Identity<scalar::F32>>;
  assert(Reduce<scalar::S32>(source, InputCast{}, optimize) == 0);
  assert(Reduce<scalar::S32>(source, expression::Identity<scalar::F32>{}, optimize) == 1);
  const auto wide = Reduce<scalar::F32>(source, expression::Identity<scalar::F32>{}, optimize);
  assert((scalar::Convert<scalar::S32, scalar::F32>(wide) == 1));
  using ScaleFirst = expression::Scale<scalar::F32>;
  using CastFirst = expression::Cast<scalar::S32, expression::Identity<scalar::F32>>;
  const auto scaled = Reduce<scalar::S32>(source,
      expression::Cast<scalar::S32, ScaleFirst>{{2.0f, {}}}, optimize);
  const auto cast = Reduce<scalar::S32>(source,
      expression::Scale<scalar::S32, CastFirst>{2, {}}, optimize);
  assert(scaled == 2 && cast == 0);
}

void CheckMixedAddition() {
  const double increment = std::ldexp(1.0, -24);
  const std::array<double, 2> source{1 + increment, 1 + increment};
  using InputCast = expression::Cast<scalar::F32, expression::Identity<scalar::F64>>;
  for (int64_t stride : {-1, 1}) {
    const ReductionInput input{0, 0, stride < 0 ? 2 : 1,
        {2}, {1}, {2}, {stride}, {false}};
    const auto record = Build(input, source.size(), 4);
    std::array<float, 4> direct{7, -1, -1, 7};
    kernels::ExecuteReductionRecord<scalar::F32>(record, source.data(), direct.data(),
        expression::Identity<scalar::F64>{});
    const auto expected = static_cast<float>(increment);
    assert((direct == std::array<float, 4>{7, expected, expected, 7}));
    std::array<float, 4> converted{7, -1, -1, 7};
    kernels::ExecuteReductionRecord<scalar::F32>(record, source.data(), converted.data(),
        InputCast{});
    assert((converted == std::array<float, 4>{7, 0, 0, 7}));
  }
  const std::array<double, 2> local_source{-1, 1 + increment};
  for (bool optimize : {false, true}) {
    assert(Reduce<scalar::F32>(local_source, expression::Identity<scalar::F64>{}, optimize) ==
        static_cast<float>(increment));
    assert(Reduce<scalar::F32>(local_source, InputCast{}, optimize) == 0);
  }
}

struct ObserveWriteback {
  using InputDtype = scalar::F64;
  using OutputDtype = scalar::F64;

  const float* destination;
  std::vector<float>* observed;

  double operator()(double value) const {
    observed->push_back(*destination);
    return value;
  }
};

void CheckWritebackBoundaries() {
  const std::array<double, 3> source{1e8, 1, -1e8};
  const ReductionInput input{0, 0, 1, {1, 3}, {1, 1}, {1, 1}, {1, 1}, {false, true}};
  for (bool optimize : {false, true}) {
    auto record = Build(input, source.size(), 3);
    if (optimize) layout::OptimizeRecordForExecution(&record);
    std::array<float, 3> result{7, 0, 7};
    std::vector<float> observed;
    kernels::ExecuteReductionRecord<scalar::F32>(record, source.data(), result.data(),
        ObserveWriteback{result.data() + 1, &observed});
    assert(result.front() == 7 && result.back() == 7);
    if (optimize) {
      assert(record.destination_strides[0] == 0);
      assert(result[1] == 1);
      assert((observed == std::vector<float>{0, 0, 0}));
    } else {
      assert(record.destination_strides[0] == 1);
      assert(result[1] == 0);
      assert((observed == std::vector<float>{0, 1e8f, 1e8f}));
    }
  }

  const ReductionInput segments{0, 0, 1, {2, 2}, {1, 2}, {1, 1}, {1, 1}, {true, true}};
  const auto record = Build(segments, 4, 3);
  const std::array<double, 4> segmented_source{1e8, 1, -1e8, 0};
  std::array<float, 3> result{7, 0, 7};
  std::vector<float> observed;
  kernels::ExecuteReductionRecord<scalar::F32>(record, segmented_source.data(), result.data(),
      ObserveWriteback{result.data() + 1, &observed});
  assert((result == std::array<float, 3>{7, 0, 7}));
  assert((observed == std::vector<float>{0, 0, 1e8f, 1e8f}));
}

template <typename Real, typename Complex>
void CheckComplex(bool optimize) {
  using Value = scalar::Value<Real>;
  const std::array<scalar::Value<Complex>, 2> source{{{1, 2}, {3, -5}}};
  const auto conjugated = Reduce<Complex>(source, expression::Conjugate<Complex>{}, optimize);
  assert((conjugated == scalar::Value<Complex>{4, 3}));
  assert(Reduce<Real>(source, expression::Conjugate<Complex>{}, optimize) == 4);
  const std::array<Value, 2> real{1, 2};
  assert((Reduce<Complex>(real, expression::Identity<Real>{}, optimize) ==
      scalar::Value<Complex>{3, 0}));

  const std::array<scalar::Value<Complex>, 1> infinite{{
      {std::numeric_limits<Value>::infinity(), 1}}};
  const auto scaled = Reduce<Complex>(infinite,
      expression::Scale<Real, expression::Identity<Complex>>{2, {}}, optimize);
  assert(std::isinf(scaled.real()) && scaled.real() > 0);
  assert(scaled.imag() == 2);
}

void CheckSpecial(bool optimize) {
  const float infinity = std::numeric_limits<float>::infinity();
  const std::array<float, 2> opposed{infinity, -infinity};
  assert(std::isnan(Reduce<scalar::F32>(opposed, expression::Identity<scalar::F32>{}, optimize)));
  const std::array<float, 1> nan{std::numeric_limits<float>::quiet_NaN()};
  assert(std::isnan(Reduce<scalar::F32>(nan, expression::Scale<scalar::F32>{0, {}}, optimize)));
  const std::array<float, 1> negative_zero{-0.0f};
  const auto zero = Reduce<scalar::F32>(negative_zero, expression::Identity<scalar::F32>{}, optimize);
  assert(zero == 0.0f && !std::signbit(zero));
}

int main() {
  CheckMixedAddition();
  CheckWritebackBoundaries();
  using Dtypes = std::tuple<scalar::Pred, scalar::S8, scalar::S16, scalar::S32,
      scalar::S64, scalar::U8, scalar::U16, scalar::U32, scalar::U64,
      scalar::F16, scalar::BF16, scalar::F32, scalar::F64, scalar::C64, scalar::C128>;
  for (bool optimize : {false, true}) {
    std::apply([&](auto... dtype) { (CheckDtype<decltype(dtype)>(optimize), ...); }, Dtypes{});
    CheckLowPrecision<scalar::F16>(2048, optimize);
    CheckLowPrecision<scalar::BF16>(256, optimize);
    CheckConversionOrder(optimize);
    CheckComplex<scalar::F32, scalar::C64>(optimize);
    CheckComplex<scalar::F64, scalar::C128>(optimize);
    CheckSpecial(optimize);
  }
}
