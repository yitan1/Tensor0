#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>

#include "numeric/scalar.inc"
#include "numeric/expression.inc"

using namespace scalar;
namespace expr = expression;

using Dtypes = std::tuple<Pred, S8, S16, S32, S64, U8, U16, U32, U64,
                          F16, BF16, F32, F64, C64, C128>;

template <typename Dtype>
void CheckBasicExpressions() {
  const auto zero = Convert<Dtype, S32>(0);
  const auto one = Convert<Dtype, S32>(1);
  const auto two = Convert<Dtype, S32>(2);
  const expr::Scale<Dtype> source_op{two};
  const expr::Scale<Dtype> base_op{one};
  const expr::AddUpdate<expr::Scale<Dtype>, expr::Scale<Dtype>> operation{source_op, base_op};
  assert(expr::Identity<Dtype>{}(two) == two);
  assert(expr::Zero<Dtype>{}(two) == zero);
  assert(expr::Conjugate<Dtype>{}(two) == Conjugate<Dtype>(two));
  assert(source_op(one) == two);
  assert(operation(one, two) == Add<Dtype>(two, two));
  assert(base_op(one) == one);
}

template <typename Source, typename... Results>
void CheckCasts(std::tuple<Results...>) {
  const auto source = Convert<Source, S32>(1);
  ((assert((expr::Cast<Results, expr::Identity<Source>>{}(source) ==
            Convert<Results, Source>(source)))), ...);
}

template <typename Source, typename Base>
void CheckCombinationTypes() {
  using BaseOp = expr::Identity<Base>;
  using Combine = expr::AddUpdate<expr::Identity<Source>, BaseOp>;
  static_assert(std::is_same_v<typename Combine::InputDtype, Source>);
  static_assert(std::is_same_v<typename Combine::OutputDtype, Promote<Source, Base>>);
  static_assert(std::is_same_v<typename Combine::BaseInputDtype, Base>);
  static_assert(std::is_invocable_v<Combine, Value<Source>, Value<Base>>);
  const auto source_term = Convert<Source, S32>(1);
  const auto original_base = Convert<Base, S32>(2);
  assert((Combine{}(source_term, original_base) == Add<Source, Base>(source_term, original_base)));
}

template <typename Source, typename... Bases>
void CheckCombinations(std::tuple<Bases...>) {
  (CheckCombinationTypes<Source, Bases>(), ...);
}

template <typename... Types>
void CheckDtypes(std::tuple<Types...> dtypes) {
  (CheckBasicExpressions<Types>(), ...);
  (CheckCasts<Types>(dtypes), ...);
  (CheckCombinations<Types>(dtypes), ...);
}

void CheckStages() {
  using Narrow = expr::Scale<F32>;
  using Wide = expr::Scale<F64, expr::Identity<F32>>;
  const float epsilon = std::ldexp(1.0F, -23);
  const float narrow_value = 1 - epsilon;
  const Narrow narrow{1 + epsilon};
  const Wide wide{2};
  const expr::AddUpdate<Narrow, Wide> narrow_source{narrow, wide};
  const expr::AddUpdate<Wide, Narrow> narrow_base{wide, narrow};
  assert((Convert<F32, F64>(narrow_source(narrow_value, -0.5F)) == 0));
  assert((Convert<F32, F64>(narrow_base(-0.5F, narrow_value)) == 0));
  const double widened_product = static_cast<double>(1 + epsilon) * narrow_value;
  assert(widened_product - 1 != 0);

  const Wide source{1 + std::ldexp(1.0, -25)};
  const expr::AddUpdate<Wide, Wide> combine{source, Wide{-1}};
  assert((Convert<F32, F64>(combine(1, 1)) == std::ldexp(1.0F, -25)));

  const expr::Scale<F16, expr::Cast<F16, expr::Identity<F32>>> half_scale{Convert<F16, F32>(1)};
  const expr::Cast<F32, decltype(half_scale)> half_round_trip{half_scale};
  const float input = 1.0003F;
  assert(half_round_trip(input) == 1);
  assert(expr::Identity<F32>{}(input) != half_round_trip(input));
  using ToHalf = expr::Cast<F16, expr::Identity<F32>>;
  const expr::AddUpdate<ToHalf, ToHalf> half_addition{};
  assert((Convert<F32, F16>(half_addition(input, -1)) == 0));

  const expr::Cast<F16, Narrow> half_output{Narrow{1.5F}};
  assert(half_output(2) == 0x4200);
  const expr::Cast<BF16, Narrow> bfloat_output{Narrow{1.5F}};
  assert(bfloat_output(2) == 0x4040);

  const expr::Scale<F32, expr::Scale<F32>> staged{2, Narrow{0.5F}};
  assert(staged(std::numeric_limits<float>::denorm_min()) == 0);
  assert(expr::Identity<F32>{}(std::numeric_limits<float>::denorm_min()) != 0);

  const expr::Scale<F16, expr::Identity<F32>> default_half_factor{Convert<F16, F32>(1)};
  static_assert(std::is_same_v<decltype(default_half_factor)::OutputDtype, F32>);
  assert(default_half_factor(input) == input);
  assert(half_round_trip(input) != default_half_factor(input));

  using WideInput = expr::Cast<F64, expr::Identity<F32>>;
  const expr::AddUpdate<WideInput, WideInput> wide_addition{};
  const expr::AddUpdate<expr::Identity<F32>, expr::Identity<F32>> narrow_addition{};
  assert(wide_addition(16777216.0F, 1.0F) == 16777217.0);
  assert((Convert<F64, F32>(narrow_addition(16777216.0F, 1.0F)) == 16777216.0));

  assert((expr::Cast<F16, expr::Identity<F32>>{}(1) == 0x3c00));
  assert((expr::Cast<F16, expr::Identity<F32>>{}(2) == 0x4000));
}

void CheckIntegers() {
  constexpr int64_t precise = (INT64_C(1) << 53) + 1;
  assert(expr::Identity<S64>{}(precise) == precise);
  assert(expr::Scale<S64>{1}(precise) == precise);
  assert((expr::AddUpdate<expr::Identity<S64>, expr::Identity<S64>>{}(precise, 1) == precise + 1));
  assert(expr::Scale<S64>{2}(INT64_MAX) == -2);
  const expr::Scale<S8, expr::Cast<S8, expr::Identity<S32>>> narrow{2};
  assert((expr::AddUpdate<decltype(narrow), expr::Identity<S32>>{narrow, {}}(127, 2) == 0));
  const expr::Cast<S8, expr::Scale<S32>> narrow_output{expr::Scale<S32>{2}};
  assert(narrow_output(127) == -2);
}

void CheckStructuralBranches() {
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  assert(expr::Zero<F32>{}(infinity) == 0);
  assert(std::isnan(expr::Scale<F32>{0}(infinity)));
  assert(std::isnan(expr::Scale<F32>{0}(nan)));
  assert(std::signbit(expr::Identity<F32>{}(-0.0F)));
  assert((!std::signbit(expr::AddUpdate<expr::Identity<F32>, expr::Identity<F32>>{}(-0.0F, 0.0F))));
  assert((std::isnan(expr::AddUpdate<expr::Identity<F32>, expr::Identity<F32>>{}(3, nan))));
  const expr::AddUpdate<expr::Identity<F32>, expr::Scale<F32>> ordinary{{}, expr::Scale<F32>{0}};
  assert(std::isnan(ordinary(1, infinity)));

  const expr::Identity<F16> half_identity{};
  const expr::Zero<F32, F16> half_zero{};
  assert(half_identity(0x7c01) == 0x7c01);
  assert(half_zero(nan) == 0);
}

void CheckComplexExpressions() {
  const expr::Scale<C64, expr::Conjugate<C64>> conjugated{{1, 2}};
  assert(conjugated({3, 4}) == std::complex<float>(11, 2));
  const expr::Cast<F32, expr::Scale<C64>> projected{expr::Scale<C64>{{1, 2}}};
  assert(projected({3, 4}) == -5);
  const expr::Cast<C64, expr::Scale<F32>> embedded{expr::Scale<F32>{2}};
  assert(embedded(3) == std::complex<float>(6, 0));

  const float infinity = std::numeric_limits<float>::infinity();
  const expr::Scale<C64, expr::Identity<F32>> real_input{{infinity, 2}};
  const expr::Scale<C64, expr::Cast<C64, expr::Identity<F32>>> complex_input{{infinity, 2}};
  assert(real_input(3).imag() == 6);
  assert(std::isnan(complex_input(3).imag()));

  const expr::AddUpdate<expr::Identity<C64>, expr::Scale<C128>> combine{{}, expr::Scale<C128>{{2, 0}}};
  assert((Convert<C64, C128>(combine({1, 2}, {3, 4})) == std::complex<float>(7, 10)));
}

struct ObservedMap {
  using InputDtype = F32;
  using OutputDtype = F32;

  int& calls;

  float operator()(float value) const {
    ++calls;
    return value;
  }
};

void CheckUpdateExecution() {
  const std::array<float, 3> source_values{1, 2, 3};
  const std::array<float, 3> base_values{-1, -2, -3};
  std::array<float, 3> output{};
  int source_calls = 0;
  int base_calls = 0;
  int source_reads = 0;
  int base_reads = 0;
  const auto read_source = [&](std::size_t index) {
    ++source_reads;
    return source_values[index];
  };
  const auto read_base = [&](std::size_t index) {
    ++base_reads;
    return base_values[index];
  };
  const expr::Scale<F64, ObservedMap> source_op{
      1 + std::ldexp(1.0, -25), ObservedMap{source_calls}};
  const expr::AddUpdate<decltype(source_op), ObservedMap> operation{source_op, ObservedMap{base_calls}};
  for (std::size_t index = 0; index < output.size(); ++index) {
    const auto combined = operation(read_source(index), read_base(index));
    output[index] = Convert<F32, decltype(operation)::OutputDtype>(combined);
    assert(output[index] == source_values[index] * std::ldexp(1.0F, -25));
  }
  assert(source_calls == 3 && base_calls == 3 && source_reads == 3 && base_reads == 3);
  const double precise_term = 1 + std::ldexp(1.0, -25);
  const auto prematurely_narrowed = Add<F32>(Convert<F32, F64>(precise_term), -1);
  assert(prematurely_narrowed == 0 && output[0] != 0);

  source_calls = base_calls = source_reads = base_reads = 0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const auto source_term = source_op(read_source(index));
    output[index] = Convert<F32, F64>(source_term);
    assert(output[index] == source_values[index]);
  }
  assert(source_calls == 3 && source_reads == 3 && base_calls == 0 && base_reads == 0);

  source_calls = base_calls = source_reads = base_reads = 0;
  const ObservedMap base_op{base_calls};
  for (std::size_t index = 0; index < output.size(); ++index) {
    output[index] = base_op(read_base(index));
    assert(output[index] == base_values[index]);
  }
  assert(source_calls == 0 && source_reads == 0 && base_calls == 3 && base_reads == 3);
}

int main() {
  using Promoted = expr::Scale<F64, expr::Identity<F32>>;
  using Rounded = expr::Cast<F16, Promoted>;
  static_assert(std::is_same_v<Promoted::InputDtype, F32>);
  static_assert(std::is_same_v<Promoted::OutputDtype, F64>);
  static_assert(std::is_same_v<Rounded::InputDtype, F32>);
  static_assert(std::is_same_v<Rounded::OutputDtype, F16>);
  static_assert(std::is_invocable_v<expr::Cast<F32>, float>);
  static_assert(!std::is_invocable_v<expr::Cast<F32>, float, float>);
  CheckDtypes(Dtypes{});
  CheckStages();
  CheckIntegers();
  CheckStructuralBranches();
  CheckComplexExpressions();
  CheckUpdateExecution();
}
