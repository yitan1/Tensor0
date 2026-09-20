#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "numeric/scalar.h"

using namespace tensor0::stride;
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>



using namespace scalar;

using Dtypes = std::tuple<Pred, S8, S16, S32, S64, U8, U16, U32, U64,
                          F16, BF16, F32, F64, C64, C128>;

template <typename Dtype>
void CheckArithmetic() {
  const auto zero = Convert<Dtype, S32>(0);
  const auto one = Convert<Dtype, S32>(1);
  const auto two = Convert<Dtype, S32>(2);
  assert(Add<Dtype>(one, one) == two);
  assert(Multiply<Dtype>(one, two) == two);
  assert(Multiply<Dtype>(zero, two) == zero);
  assert(Conjugate<Dtype>(one) == one);
  assert((Real<Dtype>(one) == Convert<Component<Dtype>, S32>(1)));
  assert(Imag<Dtype>(one) == Value<Component<Dtype>>(0));
  if constexpr (std::is_integral_v<Value<Dtype>> && !kIsHalf<Dtype> &&
                !std::is_same_v<Dtype, Pred>) {
    using Element = Value<Dtype>;
    const auto maximum = std::numeric_limits<Element>::max();
    const auto minimum = std::numeric_limits<Element>::min();
    const auto wrapped = std::is_signed_v<Element> ? minimum : Element{0};
    assert(Add<Dtype>(maximum, one) == wrapped);
    assert(Multiply<Dtype>(maximum, two) == Add<Dtype>(maximum, maximum));
    assert(Multiply<Dtype>(minimum, Convert<Dtype, S32>(-1)) == minimum);
  }
}

template <typename Source, typename... Results>
void CheckConversions(std::tuple<Results...>) {
  const auto source = Convert<Source, S32>(1);
  ((assert((Convert<Results, Source>(source) == Convert<Results, S32>(1)))), ...);
}

template <typename... Types>
void CheckDtypes(std::tuple<Types...> dtypes) {
  (CheckArithmetic<Types>(), ...);
  (CheckConversions<Types>(dtypes), ...);
}

template <typename Dtype>
void CheckIntegerConversion() {
  using Element = Value<Dtype>;
  const double infinity = std::numeric_limits<double>::infinity();
  const auto minimum = std::numeric_limits<Element>::min();
  const auto maximum = std::numeric_limits<Element>::max();
  const double upper = std::ldexp(1.0, std::numeric_limits<Element>::digits);
  assert((Convert<Dtype, F64>(infinity) == maximum));
  assert((Convert<Dtype, F64>(-infinity) == minimum));
  assert((Convert<Dtype, F64>(std::numeric_limits<double>::quiet_NaN()) == 0));
  assert((Convert<Dtype, F64>(upper) == maximum));
  assert((Convert<Dtype, F64>(std::nextafter(upper, 0.0)) ==
          static_cast<Element>(std::nextafter(upper, 0.0))));
  assert((Convert<Dtype, F64>(1.75) == 1));
  if constexpr (std::is_signed_v<Element>) {
    assert((Convert<Dtype, F64>(-1.75) == -1));
    assert((Convert<Dtype, F64>(-upper) == minimum));
  } else {
    assert((Convert<Dtype, F64>(-1.75) == 0));
  }
}

template <typename Dtype>
void CheckHalfRoundTrip(uint16_t bits) {
  const auto value = bits;
  assert((Convert<Dtype, Dtype>(value) == bits));
  const auto decoded = Convert<F32, Dtype>(value);
  const auto encoded = Convert<Dtype, F32>(decoded);
  const auto expected = std::is_same_v<Dtype, BF16> && std::isnan(decoded)
      ? static_cast<uint16_t>(bits | 0x0040U) : bits;
  assert(encoded == expected);
}

void CheckHalf() {
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    CheckHalfRoundTrip<F16>(static_cast<uint16_t>(bits));
    CheckHalfRoundTrip<BF16>(static_cast<uint16_t>(bits));
  }
  assert((Convert<F16, F32>(1.00048828125F) == 0x3c00));
  assert((Convert<F16, F32>(1.00146484375F) == 0x3c02));
  assert((Convert<BF16, F32>(1.00390625F) == 0x3f80));
  assert((Convert<BF16, F32>(1.01171875F) == 0x3f82));
  assert((Convert<F16, F32>(-0.0F) == 0x8000));
  assert((Convert<BF16, F32>(-0.0F) == 0x8000));
  assert((Convert<F16, F32>(65520.0F) == 0x7c00));
  assert((Convert<F16, F32>(std::ldexp(1.0F, -24)) == 1));
  assert((Convert<F16, F32>(std::ldexp(1.0F, -25)) == 0));
  assert(Add<F16>(0x3e00, 0x4000) == 0x4300);
  assert(Multiply<BF16>(0x3fc0, 0x4000) == 0x4040);
  assert(std::isnan(Convert<F32, F16>(Multiply<F16>(0, 0x7c00))));
  assert(std::isnan(Convert<F32, BF16>(Multiply<BF16>(0, 0x7f80))));
}

template <typename Dtype>
void CheckComplex() {
  using Element = Value<Component<Dtype>>;
  const auto infinity = std::numeric_limits<Element>::infinity();
  const auto nan = std::numeric_limits<Element>::quiet_NaN();
  assert(Multiply<Dtype>({1, 2}, Value<Dtype>{3, 4}) == Value<Dtype>(-5, 10));
  assert((Multiply<Dtype, Component<Dtype>>({1, 2}, Element{3}) == Value<Dtype>(3, 6)));
  const auto scaled = Multiply<Dtype, Component<Dtype>>({1, 2}, infinity);
  assert(scaled.real() == infinity && scaled.imag() == infinity);
  const auto product = Multiply<Dtype>({1, 2}, Value<Dtype>{infinity, 0});
  assert(product.real() == infinity && product.imag() == infinity);
  assert((Multiply<Dtype, Component<Dtype>>({infinity, 2}, Element{3}).imag() == 6));
  assert(std::isnan(Multiply<Dtype>({infinity, 2}, Value<Dtype>{3, 0}).imag()));
  const auto epsilon = std::numeric_limits<Element>::epsilon();
  const auto rounded = Multiply<Dtype>({1 + epsilon, 1},
                                       Value<Dtype>{1 - epsilon, 1});
  assert(rounded.real() == -(epsilon * epsilon));
  assert(std::isnan(Multiply<Dtype>({0, 0}, Value<Dtype>{infinity, 0}).real()));
  assert(std::isnan(Multiply<Dtype>({0, 0}, Value<Dtype>{nan, 0}).real()));
  assert(Add<Dtype>({1, 2}, {3, 4}) == Value<Dtype>(4, 6));
  assert(Conjugate<Dtype>({1, 2}) == Value<Dtype>(1, -2));
  assert(std::signbit(Imag<Dtype>(Conjugate<Dtype>({1, 0}))));
  assert((Convert<Pred, Dtype>({0, 1})));
  assert((Convert<S32, Dtype>({3.75, 9}) == 3));
  assert((Convert<Component<Dtype>, Dtype>({3.75, 9}) == Element{3.75}));
  assert(Real<Dtype>({3, 4}) == 3 && Imag<Dtype>({3, 4}) == 4);
}

int main() {
  static_assert(std::is_same_v<Value<F16>, uint16_t>);
  static_assert(std::is_same_v<Value<BF16>, uint16_t>);
  static_assert(sizeof(Value<F16>) == 2 && sizeof(Value<BF16>) == 2);
  static_assert(!std::is_same_v<F16, BF16>);
  static_assert(sizeof(Value<S64>) == 8);
  static_assert(sizeof(Value<C128>) == 16);
  CheckDtypes(Dtypes{});
  CheckIntegerConversion<S8>();
  CheckIntegerConversion<S16>();
  CheckIntegerConversion<S32>();
  CheckIntegerConversion<S64>();
  CheckIntegerConversion<U8>();
  CheckIntegerConversion<U16>();
  CheckIntegerConversion<U32>();
  CheckIntegerConversion<U64>();
  CheckHalf();
  CheckComplex<C64>();
  CheckComplex<C128>();
  assert((Convert<S8, U64>(UINT64_MAX) == -1));
  assert((Convert<U64, S8>(-1) == UINT64_MAX));
  assert((Convert<S64, U64>(UINT64_MAX) == -1));
  assert((Convert<U64, S64>(INT64_MIN) == (UINT64_C(1) << 63)));
  assert((Convert<U64, U64>(UINT64_MAX) == UINT64_MAX));
  assert((Convert<S64, S64>(INT64_MIN) == INT64_MIN));
  assert(std::signbit(Real<F32>(-0.0F)));
  assert(std::signbit(Conjugate<F64>(-0.0)));
  assert(std::isnan(Multiply<F32>(0, std::numeric_limits<float>::infinity())));
  assert(std::isnan(Multiply<F64>(0, std::numeric_limits<double>::quiet_NaN())));
}
