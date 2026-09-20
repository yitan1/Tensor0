#pragma once

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tensor0::stride::scalar {

struct Pred { using Type = bool; };
struct S8 { using Type = int8_t; };
struct S16 { using Type = int16_t; };
struct S32 { using Type = int32_t; };
struct S64 { using Type = int64_t; };
struct U8 { using Type = uint8_t; };
struct U16 { using Type = uint16_t; };
struct U32 { using Type = uint32_t; };
struct U64 { using Type = uint64_t; };
struct F16 { using Type = uint16_t; };
struct BF16 { using Type = uint16_t; };
struct F32 { using Type = float; };
struct F64 { using Type = double; };
struct C64 { using Type = std::complex<float>; };
struct C128 { using Type = std::complex<double>; };

template <typename Dtype>
using Value = typename Dtype::Type;

template <typename Dtype>
constexpr bool kIsComplex = std::is_same_v<Dtype, C64> ||
                           std::is_same_v<Dtype, C128>;

template <typename Dtype>
constexpr bool kIsHalf = std::is_same_v<Dtype, F16> ||
                        std::is_same_v<Dtype, BF16>;

template <typename Dtype>
using Component = std::conditional_t<std::is_same_v<Dtype, C64>, F32,
                  std::conditional_t<std::is_same_v<Dtype, C128>, F64, Dtype>>;

namespace detail {

template <typename Left, typename Right>
constexpr auto PromoteTypes() {
  if constexpr (std::is_same_v<Left, Right> || std::is_same_v<Right, Pred>) {
    return std::type_identity<Left>{};
  } else if constexpr (std::is_same_v<Left, Pred>) {
    return std::type_identity<Right>{};
  } else if constexpr (kIsComplex<Left> || kIsComplex<Right>) {
    constexpr bool wide = std::is_same_v<Left, C128> || std::is_same_v<Right, C128> ||
                          std::is_same_v<Left, F64> || std::is_same_v<Right, F64>;
    return std::type_identity<std::conditional_t<wide, C128, C64>>{};
  } else if constexpr (std::is_same_v<Left, F64> || std::is_same_v<Right, F64>) {
    return std::type_identity<F64>{};
  } else if constexpr (std::is_same_v<Left, F32> || std::is_same_v<Right, F32> ||
                       (kIsHalf<Left> && kIsHalf<Right>)) {
    return std::type_identity<F32>{};
  } else if constexpr (kIsHalf<Left>) {
    return std::type_identity<Left>{};
  } else if constexpr (kIsHalf<Right>) {
    return std::type_identity<Right>{};
  } else if constexpr (std::is_signed_v<Value<Left>> == std::is_signed_v<Value<Right>>) {
    return std::type_identity<std::conditional_t<
        (sizeof(Value<Left>) >= sizeof(Value<Right>)), Left, Right>>{};
  } else {
    using Signed = std::conditional_t<std::is_signed_v<Value<Left>>, Left, Right>;
    using Unsigned = std::conditional_t<std::is_signed_v<Value<Left>>, Right, Left>;
    if constexpr (sizeof(Value<Signed>) > sizeof(Value<Unsigned>)) {
      return std::type_identity<Signed>{};
    } else {
      constexpr auto width = sizeof(Value<Unsigned>);
      return std::type_identity<std::conditional_t<width == 1, S16,
          std::conditional_t<width == 2, S32,
          std::conditional_t<width == 4, S64, F64>>>>{};
    }
  }
}

inline float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
  uint32_t exponent = (value >> 10) & 0x1FU;
  uint32_t mantissa = value & 0x03FFU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      uint32_t float_exponent = 113;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        --float_exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | (float_exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    exponent += 112;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

inline uint16_t FloatToHalf(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
  const uint32_t exponent = (bits >> 23) & 0xFFU;
  const uint32_t mantissa = bits & 0x007FFFFFU;
  if (exponent == 0xFFU) {
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7C00U);
    }
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
    if (payload == 0) {
      payload = 1;
    }
    return static_cast<uint16_t>(sign | 0x7C00U | payload);
  }

  int32_t half_exponent = static_cast<int32_t>(exponent) - 112;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const uint32_t significand = mantissa | 0x00800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = significand >> shift;
    const uint32_t remainder = significand & ((UINT32_C(1) << shift) - 1);
    const uint32_t halfway = UINT32_C(1) << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_mantissa & 1U) != 0)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x0400U) {
      rounded_mantissa = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
      }
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint16_t>(half_exponent) << 10) |
      static_cast<uint16_t>(rounded_mantissa));
}

inline float BFloat16ToFloat(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

inline uint16_t FloatToBFloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7F800000U) == 0x7F800000U &&
      (bits & 0x007FFFFFU) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040U);
  }
  bits += 0x00007FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

template <typename Real>
Real FullComplexProductReal(Real left_real, Real left_imaginary,
                            Real right_real, Real right_imaginary) {
  return std::fma(left_real, right_real,
                  -(left_imaginary * right_imaginary));
}

template <typename Real>
std::complex<Real> FullComplexMultiply(Real left_real, Real left_imaginary,
                                       Real right_real,
                                       Real right_imaginary) {
  return {FullComplexProductReal(left_real, left_imaginary, right_real,
                                 right_imaginary),
          std::fma(left_imaginary, right_real,
                   left_real * right_imaginary)};
}

template <typename Element>
uint64_t IntegerBits(Element value) {
  return static_cast<std::make_unsigned_t<Element>>(value);
}

}

template <typename Result, typename Source>
Value<Result> Convert(Value<Source> value) {
  using Output = Value<Result>;
  if constexpr (std::is_same_v<Source, Result>) {
    return value;
  } else if constexpr (std::is_same_v<Source, F16>) {
    return Convert<Result, F32>(detail::HalfToFloat(value));
  } else if constexpr (std::is_same_v<Source, BF16>) {
    return Convert<Result, F32>(detail::BFloat16ToFloat(value));
  } else if constexpr (std::is_same_v<Result, Pred>) {
    if constexpr (kIsComplex<Source>) {
      return value.real() != 0 || value.imag() != 0;
    } else {
      return value != 0;
    }
  } else if constexpr (kIsComplex<Source>) {
    if constexpr (kIsComplex<Result>) {
      return {Convert<Component<Result>, Component<Source>>(value.real()),
              Convert<Component<Result>, Component<Source>>(value.imag())};
    } else {
      return Convert<Result, Component<Source>>(value.real());
    }
  } else if constexpr (std::is_same_v<Result, F16>) {
    return detail::FloatToHalf(Convert<F32, Source>(value));
  } else if constexpr (std::is_same_v<Result, BF16>) {
    return detail::FloatToBFloat16(Convert<F32, Source>(value));
  } else if constexpr (kIsComplex<Result>) {
    return {Convert<Component<Result>, Source>(value), 0};
  } else if constexpr (std::is_integral_v<Output>) {
    if constexpr (std::is_integral_v<Value<Source>>) {
      return static_cast<Output>(value);
    } else {
      const long double upper =
          std::ldexp(1.0L, std::numeric_limits<Output>::digits);
      const long double lower = std::is_signed_v<Output> ? -upper : 0;
      if (std::isnan(value)) return 0;
      if (value >= upper) return std::numeric_limits<Output>::max();
      if (value <= lower) return std::numeric_limits<Output>::min();
      return static_cast<Output>(value);
    }
  } else {
    return static_cast<Output>(value);
  }
}

template <typename Left, typename Right>
using Promote = typename decltype(detail::PromoteTypes<Left, Right>())::type;

template <typename Left, typename Right = Left>
Value<Promote<Left, Right>> Add(Value<Left> left_value, Value<Right> right_value) {
  using Dtype = Promote<Left, Right>;
  const auto left = Convert<Dtype, Left>(left_value);
  const auto right = Convert<Dtype, Right>(right_value);
  if constexpr (kIsHalf<Dtype>) {
    return Convert<Dtype, F32>(Add<F32>(Convert<F32, Dtype>(left),
                                       Convert<F32, Dtype>(right)));
  } else if constexpr (std::is_same_v<Dtype, Pred>) {
    return left || right;
  } else if constexpr (std::is_integral_v<Value<Dtype>>) {
    return static_cast<Value<Dtype>>(
        detail::IntegerBits(left) + detail::IntegerBits(right));
  } else if constexpr (kIsComplex<Dtype>) {
    return {Add<Component<Dtype>>(left.real(), right.real()),
            Add<Component<Dtype>>(left.imag(), right.imag())};
  } else {
    return left + right;
  }
}

template <typename Left, typename Right = Left>
Value<Promote<Left, Right>> Multiply(Value<Left> left_value, Value<Right> right_value) {
  using Dtype = Promote<Left, Right>;
  if constexpr (kIsComplex<Left> != kIsComplex<Right>) {
    using Real = Component<Dtype>;
    if constexpr (kIsComplex<Left>) {
      const auto left = Convert<Dtype, Left>(left_value);
      const auto right = Convert<Real, Right>(right_value);
      return {Multiply<Real>(left.real(), right), Multiply<Real>(left.imag(), right)};
    } else {
      const auto left = Convert<Real, Left>(left_value);
      const auto right = Convert<Dtype, Right>(right_value);
      return {Multiply<Real>(left, right.real()), Multiply<Real>(left, right.imag())};
    }
  } else {
    const auto left = Convert<Dtype, Left>(left_value);
    const auto right = Convert<Dtype, Right>(right_value);
    if constexpr (kIsHalf<Dtype>) {
      return Convert<Dtype, F32>(Multiply<F32>(Convert<F32, Dtype>(left),
                                              Convert<F32, Dtype>(right)));
    } else if constexpr (std::is_same_v<Dtype, Pred>) {
      return left && right;
    } else if constexpr (std::is_integral_v<Value<Dtype>>) {
      return static_cast<Value<Dtype>>(
          detail::IntegerBits(left) * detail::IntegerBits(right));
    } else if constexpr (kIsComplex<Dtype>) {
      return detail::FullComplexMultiply(left.real(), left.imag(),
                                         right.real(), right.imag());
    } else {
      return left * right;
    }
  }
}

template <typename Dtype>
Value<Component<Dtype>> Real(Value<Dtype> value) {
  if constexpr (kIsComplex<Dtype>) return value.real();
  else return value;
}

template <typename Dtype>
Value<Component<Dtype>> Imag(Value<Dtype> value) {
  if constexpr (kIsComplex<Dtype>) return value.imag();
  else return Value<Dtype>(0);
}

template <typename Dtype>
Value<Dtype> Conjugate(Value<Dtype> value) {
  if constexpr (kIsComplex<Dtype>) return {Real<Dtype>(value), -Imag<Dtype>(value)};
  else return value;
}

template <typename Dtype>
bool IsZero(Value<Dtype> value) {
  if constexpr (kIsHalf<Dtype>) return Convert<F32, Dtype>(value) == 0.0F;
  else return value == Value<Dtype>{};
}

template <typename Dtype>
bool IsOne(Value<Dtype> value) {
  if constexpr (kIsHalf<Dtype>) return Convert<F32, Dtype>(value) == 1.0F;
  else return value == Convert<Dtype, S32>(1);
}

}
