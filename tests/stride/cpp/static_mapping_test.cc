#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>

#include "numeric/scalar.inc"
#include "numeric/expression.inc"

using namespace scalar;
namespace expr = expression;

using Dtypes = std::tuple<Pred, S8, S16, S32, S64, U8, U16, U32, U64,
                          F16, BF16, F32, F64, C64, C128>;

template <typename Source, typename Result>
void CheckIdentity() {
  const expr::Cast<Result, expr::Identity<Source>> operation{};
  assert((operation(Convert<Source, S32>(1)) == Convert<Result, S32>(1)));
}

template <typename Source, typename... Results>
void CheckSource(std::tuple<Results...>) {
  (CheckIdentity<Source, Results>(), ...);
  const expr::Scale<Source> operation{Convert<Source, S32>(2)};
  assert((operation(Convert<Source, S32>(1)) == Convert<Source, S32>(2)));
}

template <typename... Types>
void CheckDtypes(std::tuple<Types...> dtypes) {
  (CheckSource<Types>(dtypes), ...);
}

void CheckCoefficientBinding() {
  double coefficient = 1.00048828125;
  using HalfScale = expr::Scale<F16, expr::Cast<F16, expr::Identity<F32>>>;
  const expr::Cast<F32, HalfScale> half_operation{
      HalfScale{Convert<F16, F64>(coefficient), {}}};
  coefficient = 2;
  for (float value : {1.0F, 2.0F, 3.0F}) assert(half_operation(value) == value);
  const double precise_factor = 1 + std::ldexp(1.0, -40);
  const expr::Scale<F64, expr::Identity<F32>> precise_operation{precise_factor};
  assert(precise_operation(1) == precise_factor);
  assert(expr::Identity<F32>{}(3) == 3);
  assert(std::signbit(expr::Identity<F32>{}(-0.0F)));
  assert(expr::Identity<F16>{}(0x7c01) == 0x7c01);
  const expr::Scale<F16> nan_operation{0x7e55};
  assert(std::isnan(Convert<F32, F16>(nan_operation(0x7d60))));
  assert(expr::Identity<F16>{}(0x7d60) == 0x7d60);
}

void CheckNoImplicitShortcuts() {
  const float infinity = std::numeric_limits<float>::infinity();
  const expr::Scale<F32> zero_operation{0};
  assert(std::isnan(zero_operation(infinity)));
  using HalfScale = expr::Scale<F16, expr::Cast<F16, expr::Identity<F32>>>;
  const expr::Cast<F32, HalfScale> half_operation{
      HalfScale{Convert<F16, F32>(1), {}}};
  assert(half_operation(1.0003F) == 1);
  const expr::Cast<F32, expr::Cast<F16, expr::Identity<F32>>> half_round_trip{};
  assert(half_round_trip(1.0003F) == 1);
  constexpr int64_t precise = (INT64_C(1) << 53) + 1;
  assert(expr::Identity<S64>{}(precise) == precise);
  using WideScale = expr::Scale<F64, expr::Identity<S64>>;
  const expr::Cast<S64, WideScale> wide_operation{WideScale{1}};
  assert(wide_operation(precise) == precise - 1);
}

void CheckMappingStages() {
  const expr::Cast<F16, expr::Scale<F32>> half_output{expr::Scale<F32>{1.0003F}};
  assert(half_output(1.0003F) == 0x3c01);
  const expr::Scale<F32, expr::Cast<F32, expr::Identity<C64>>> projected_input{2};
  assert(projected_input({3, 4}) == 6);
  const expr::Cast<F32, expr::Scale<C64>> projected_output{expr::Scale<C64>{{1, 2}}};
  assert(projected_output({3, 4}) == -5);
  const float infinity = std::numeric_limits<float>::infinity();
  const expr::Cast<C64, expr::Scale<F32>> embedded_output{expr::Scale<F32>{infinity}};
  assert(embedded_output(3).real() == infinity && embedded_output(3).imag() == 0);
  const expr::Scale<C64, expr::Cast<C64, expr::Identity<F32>>> complex_input{{infinity, 2}};
  assert(std::isnan(complex_input(3).imag()));
  const expr::Scale<C64, expr::Identity<F32>> real_input{{infinity, 2}};
  assert(real_input(3).imag() == 6);
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

void CheckInputMap() {
  int calls = 0;
  const expr::Scale<F32, ObservedMap> operation{2, ObservedMap{calls}};
  assert(calls == 0);
  assert(operation(3) == 6);
  assert(calls == 1);
  assert(operation(4) == 8);
  assert(calls == 2);
  calls = 0;
  const ObservedMap unscaled{calls};
  assert(unscaled(3) == 3 && calls == 1);
  assert(unscaled(4) == 4 && calls == 2);
  const expr::Conjugate<C64> conjugate{};
  assert(conjugate({3, 4}) == std::complex<float>(3, -4));
  const expr::Cast<F32, expr::Conjugate<C64>> projected{};
  assert(projected({3, 4}) == 3);
  const expr::Scale<F32> explicit_scale{2};
  assert(explicit_scale(3) == 6);
  using HalfInput = expr::Cast<F16, expr::Identity<F32>>;
  const expr::Cast<F32, HalfInput> half_round_trip{};
  assert(half_round_trip(1.0003F) == 1);
  assert(std::isinf(half_round_trip(65520.0F)));
  using NarrowInput = expr::Cast<F32, expr::Identity<F64>>;
  const expr::Cast<F64, NarrowInput> narrow_round_trip{};
  assert(narrow_round_trip(16777217.0) == 16777216.0);
}

void CheckExplicitMapping() {
  const expr::Identity<F16> identity{};
  static_assert(std::is_same_v<std::remove_cv_t<decltype(identity)>, expr::Identity<F16>>);
  assert(identity(0x7d60) == 0x7d60);
  const expr::Scale<F16> unit{0x3c00};
  assert((unit(0x7d60) & 0x0200) != 0);
  const expr::Scale<F32> zero{0};
  assert(std::isnan(zero(std::numeric_limits<float>::infinity())));

  const expr::Cast<F16, expr::Scale<F16, expr::Cast<F16, expr::Identity<F32>>>>
      transpose{{0x3c01, {}}};
  static_assert(std::is_same_v<typename decltype(transpose)::InputDtype, F32>);
  static_assert(std::is_same_v<typename decltype(transpose)::OutputDtype, F16>);
  assert(transpose(1.00048828125F) == 0x3c01);
  const expr::Scale<F16, expr::Identity<F32>> mixed{0x3c01};
  static_assert(std::is_same_v<typename decltype(mixed)::OutputDtype, F32>);
  const expr::Cast<F16, decltype(mixed)> late_cast{mixed};
  assert(late_cast(1.00048828125F) == 0x3c02);

  const expr::Scale<C64> complex_transpose{{0, 1}};
  assert(complex_transpose({1, 1}) == std::complex<float>(-1, 1));
  const expr::Cast<C64, expr::Scale<F32>> embedded{{std::numeric_limits<float>::infinity(), {}}};
  assert(std::isinf(embedded(3).real()) && embedded(3).imag() == 0);
  constexpr int64_t precise = (INT64_C(1) << 53) + 1;
  assert((expr::Identity<S64>{}(precise) == precise));
  const expr::Cast<S64, expr::Scale<F64, expr::Identity<S64>>> rounded{{1, {}}};
  assert(rounded(precise) == precise - 1);
}

int main() {
  CheckDtypes(Dtypes{});
  CheckCoefficientBinding();
  CheckNoImplicitShortcuts();
  CheckMappingStages();
  CheckInputMap();
  CheckExplicitMapping();
}
