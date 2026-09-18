#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>

#include "numeric/scalar.inc"

using namespace scalar;
using Dtypes = std::tuple<Pred, S8, S16, S32, S64, U8, U16, U32, U64,
                          F16, BF16, F32, F64, C64, C128>;

template <typename Dtype, std::size_t Index = 0>
constexpr std::size_t DtypeIndex() {
  if constexpr (std::is_same_v<Dtype, std::tuple_element_t<Index, Dtypes>>) return Index;
  else return DtypeIndex<Dtype, Index + 1>();
}

template <typename Dtype>
Value<Dtype> Sample(double real, double imaginary) {
  if constexpr (kIsComplex<Dtype>) {
    return {Convert<Component<Dtype>, F64>(real),
            Convert<Component<Dtype>, F64>(imaginary)};
  } else {
    return Convert<Dtype, F64>(real);
  }
}

template <typename Dtype>
void Print(Value<Dtype> value) {
  std::cout << ' ' << Convert<F64, Component<Dtype>>(Real<Dtype>(value))
            << ' ' << Convert<F64, Component<Dtype>>(Imag<Dtype>(value));
}

template <typename Left, typename Right>
void CheckPair() {
  using Result = Promote<Left, Right>;
  static_assert(std::is_same_v<Result, Promote<Right, Left>>);
  static_assert(std::is_same_v<decltype(Add<Left, Right>({}, {})), Value<Result>>);
  static_assert(std::is_same_v<decltype(Multiply<Left, Right>({}, {})), Value<Result>>);
  constexpr std::array<double, 6> real = {-2.5, 0, 1.25, 7.5, 1.0003, 127};
  constexpr std::array<double, 6> imaginary = {0.25, -0.5, 1.5, 3, -2, 0};
  std::cout << DtypeIndex<Left>() << ' ' << DtypeIndex<Right>() << ' ' << DtypeIndex<Result>();
  for (std::size_t index = 0; index < real.size(); ++index) {
    const auto left = Sample<Left>(real[index], imaginary[index]);
    const auto right = Sample<Right>(real[real.size() - 1 - index], imaginary[index]);
    Print<Result>(Add<Left, Right>(left, right));
    Print<Result>(Multiply<Left, Right>(left, right));
  }
  std::cout << '\n';
}

template <typename Left, typename... Rights>
void CheckRow(std::tuple<Rights...>) {
  (CheckPair<Left, Rights>(), ...);
}

template <typename... Types>
void CheckTable(std::tuple<Types...> types) {
  (CheckRow<Types>(types), ...);
}

void CheckBoundaries() {
  static_assert(std::is_same_v<Promote<S64, F16>, F16>);
  static_assert(std::is_same_v<Promote<F16, BF16>, F32>);
  static_assert(std::is_same_v<Promote<S32, U32>, S64>);
  static_assert(std::is_same_v<Promote<S64, U64>, F64>);
  static_assert(std::is_same_v<Promote<C64, S64>, C64>);
  static_assert(std::is_same_v<Promote<C64, F64>, C128>);
  assert((Add<S8, U8>(-1, 255) == 254));
  assert((Add<S32, U32>(-1, UINT32_MAX) == INT64_C(4294967294)));
  assert((Multiply<S64, U32>(INT64_MAX, 2) == -2));
  assert((Add<S64, U64>(-1, UINT64_MAX) == std::ldexp(1.0, 64)));
  assert((Multiply<Pred, S8>(true, -7) == -7));
  assert((Add<F16, BF16>(0x3c00, 0x3f80) == 2.0F));
  assert((Multiply<F16, S32>(0x3c00, 65504) == 0x7bff));
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  assert((Multiply<C64, F32>({infinity, 2}, 3).imag() == 6));
  assert((Multiply<F32, C64>(3, {infinity, 2}).imag() == 6));
  assert(std::isnan(Multiply<C64>({infinity, 2}, Convert<C64, F32>(3)).imag()));
  assert((std::isnan(Multiply<C64, F32>({nan, 2}, 3).real())));
  assert((Multiply<C64, F32>({nan, 2}, 3).imag() == 6));
  assert((std::signbit(Multiply<C64, F32>({2, -0.0F}, 3).imag())));
  assert((std::isnan(Multiply<F16, F32>(0, infinity))));
  const auto wide = Multiply<C128, F64>(Convert<C128, C64>({1, 2}), 1 + std::ldexp(1.0, -30));
  const auto narrow = Convert<C128, C64>(Multiply<C64, F32>({1, 2}, 1));
  assert(wide.real() != narrow.real());
}

int main() {
  CheckBoundaries();
  std::cout << std::setprecision(17);
  CheckTable(Dtypes{});
}
