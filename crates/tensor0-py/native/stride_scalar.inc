template <typename To, typename From>
To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float HalfToFloat(uint16_t value) {
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
  return BitCast<float>(bits);
}

uint16_t FloatToHalf(float value) {
  const uint32_t bits = BitCast<uint32_t>(value);
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

float BFloat16ToFloat(uint16_t value) {
  return BitCast<float>(static_cast<uint32_t>(value) << 16);
}

uint16_t FloatToBFloat16(float value) {
  uint32_t bits = BitCast<uint32_t>(value);
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

template <typename Real>
std::complex<Real> PromotedComplexMultiply(std::complex<Real> scale,
                                           Real value) {
  return {scale.real() * value, scale.imag() * value};
}

template <typename Element>
Element AddInteger(Element left, Element right) {
  using Unsigned = std::make_unsigned_t<Element>;
  const Unsigned sum = static_cast<Unsigned>(
      static_cast<uint64_t>(BitCast<Unsigned>(left)) +
      static_cast<uint64_t>(BitCast<Unsigned>(right)));
  return BitCast<Element>(sum);
}

struct IdentityScalar {
  template <typename RecordType>
  IdentityScalar Bind(const RecordType&) const { return {}; }

  template <typename Element>
  Element operator()(Element value) const { return value; }
};

template <typename Element>
struct ZeroScalar {
  template <typename Input>
  Element operator()(Input) const { return Element{}; }
};

template <typename Value, typename Scale, typename Map = IdentityScalar>
struct BoundScalarScale {
  Value factor;
  Scale scale;
  Map map{};

  template <typename Input>
  auto operator()(Input value) const {
    return scale(factor, map(value));
  }
};

struct ScalarValue {
  long double real = 0;
  long double imaginary = 0;
  bool integral = false;
};

template <typename Value>
Value LoadScalar(const void* address) {
  Value value;
  std::memcpy(&value, address, sizeof(Value));
  return value;
}

ScalarValue ReadScalar(const void* address, uint64_t dtype) {
  switch (dtype) {
    case kDtypeF16:
      return {HalfToFloat(LoadScalar<uint16_t>(address)), 0, false};
    case kDtypeBF16:
      return {BFloat16ToFloat(LoadScalar<uint16_t>(address)), 0, false};
    case kDtypeF32:
      return {LoadScalar<float>(address), 0, false};
    case kDtypeF64:
      return {LoadScalar<double>(address), 0, false};
    case kDtypeC64: {
      const auto value = LoadScalar<std::complex<float>>(address);
      return {value.real(), value.imag(), false};
    }
    case kDtypeC128: {
      const auto value = LoadScalar<std::complex<double>>(address);
      return {value.real(), value.imag(), false};
    }
    case kDtypePred:
      return {LoadScalar<bool>(address) ? 1.0L : 0.0L, 0, true};
    case kDtypeS8:
      return {static_cast<long double>(LoadScalar<int8_t>(address)), 0, true};
    case kDtypeS16:
      return {static_cast<long double>(LoadScalar<int16_t>(address)), 0, true};
    case kDtypeS32:
      return {static_cast<long double>(LoadScalar<int32_t>(address)), 0, true};
    case kDtypeS64:
      return {static_cast<long double>(LoadScalar<int64_t>(address)), 0, true};
    case kDtypeU8:
      return {static_cast<long double>(LoadScalar<uint8_t>(address)), 0, true};
    case kDtypeU16:
      return {static_cast<long double>(LoadScalar<uint16_t>(address)), 0, true};
    case kDtypeU32:
      return {static_cast<long double>(LoadScalar<uint32_t>(address)), 0, true};
    case kDtypeU64:
      return {static_cast<long double>(LoadScalar<uint64_t>(address)), 0, true};
    default:
      throw std::invalid_argument("unsupported scalar dtype");
  }
}

bool SignedScalarDtype(uint64_t dtype) {
  return dtype == kDtypeS8 || dtype == kDtypeS16 ||
         dtype == kDtypeS32 || dtype == kDtypeS64;
}

bool IntegralScalarDtype(uint64_t dtype) {
  return dtype == kDtypePred || SignedScalarDtype(dtype) || dtype >= kDtypeU8;
}

uint64_t ScalarIntegerBits(ScalarValue value) {
  return value.real < 0
             ? static_cast<uint64_t>(static_cast<int64_t>(value.real))
             : static_cast<uint64_t>(value.real);
}

uint64_t ScalarIntegerBits(ScalarValue value, uint64_t dtype) {
  const unsigned width = static_cast<unsigned>(DtypeItemSize(dtype) * 8);
  const bool signed_result = SignedScalarDtype(dtype);
  uint64_t bits;
  if (value.integral) {
    bits = ScalarIntegerBits(value);
  } else {
    const long double upper = std::ldexp(1.0L, signed_result ? width - 1 : width);
    const long double lower = signed_result ? -upper : 0;
    if (std::isnan(value.real)) {
      bits = 0;
    } else if (value.real >= upper) {
      bits = signed_result ? (UINT64_MAX >> (65 - width))
                           : (UINT64_MAX >> (64 - width));
    } else if (value.real <= lower) {
      bits = signed_result ? (UINT64_C(1) << (width - 1)) : 0;
    } else {
      bits = ScalarIntegerBits(value);
    }
  }
  if (width < 64) {
    bits &= (UINT64_C(1) << width) - 1;
    if (signed_result && (bits & (UINT64_C(1) << (width - 1)))) {
      bits |= UINT64_MAX << width;
    }
  }
  return bits;
}

template <ffi::DataType Dtype>
constexpr bool kFloatingScalarDtype = Dtype == ffi::F16 || Dtype == ffi::BF16 ||
    Dtype == ffi::F32 || Dtype == ffi::F64 || Dtype == ffi::C64 || Dtype == ffi::C128;

template <ffi::DataType Dtype>
ffi::NativeType<Dtype> ConvertScalar(ScalarValue value);

template <ffi::DataType Result, ffi::DataType Source>
ffi::NativeType<Result> ConvertScalar(ffi::NativeType<Source> value) {
  if constexpr (Source == Result) {
    return value;
  } else if constexpr (kFloatingScalarDtype<Source> && kFloatingScalarDtype<Result>) {
    if constexpr (Source == ffi::F16) {
      return ConvertScalar<Result, ffi::F32>(HalfToFloat(value));
    } else if constexpr (Source == ffi::BF16) {
      return ConvertScalar<Result, ffi::F32>(BFloat16ToFloat(value));
    } else if constexpr (Source == ffi::C64 || Source == ffi::C128) {
      constexpr auto component = Source == ffi::C64 ? ffi::F32 : ffi::F64;
      if constexpr (Result == ffi::C64 || Result == ffi::C128) {
        constexpr auto target_component = Result == ffi::C64 ? ffi::F32 : ffi::F64;
        return {ConvertScalar<target_component, component>(value.real()),
                ConvertScalar<target_component, component>(value.imag())};
      } else {
        return ConvertScalar<Result, component>(value.real());
      }
    } else if constexpr (Result == ffi::F16) {
      return FloatToHalf(ConvertScalar<ffi::F32, Source>(value));
    } else if constexpr (Result == ffi::BF16) {
      return FloatToBFloat16(ConvertScalar<ffi::F32, Source>(value));
    } else if constexpr (Result == ffi::C64 || Result == ffi::C128) {
      constexpr auto component = Result == ffi::C64 ? ffi::F32 : ffi::F64;
      return {ConvertScalar<component, Source>(value), 0};
    } else {
      return static_cast<ffi::NativeType<Result>>(value);
    }
  } else {
    return ConvertScalar<Result>(ReadScalar(&value, DescriptorDtypeCode<Source>()));
  }
}

template <ffi::DataType Dtype>
ffi::NativeType<Dtype> ConvertScalar(ScalarValue value) {
  using Native = ffi::NativeType<Dtype>;
  if constexpr (Dtype == ffi::PRED) {
    return value.real != 0 || value.imaginary != 0;
  } else if constexpr (Dtype == ffi::F16 || Dtype == ffi::BF16) {
    return ConvertScalar<Dtype, ffi::F32>(static_cast<float>(value.real));
  } else if constexpr (Dtype == ffi::F32 || Dtype == ffi::F64) {
    return static_cast<Native>(value.real);
  } else if constexpr (Dtype == ffi::C64 || Dtype == ffi::C128) {
    using Real = typename Native::value_type;
    return {static_cast<Real>(value.real), static_cast<Real>(value.imaginary)};
  } else if constexpr (std::is_integral_v<Native>) {
    using Unsigned = std::make_unsigned_t<Native>;
    const auto bits = ScalarIntegerBits(value, DescriptorDtypeCode<Dtype>());
    return BitCast<Native>(static_cast<Unsigned>(bits));
  } else {
    static_assert(Dtype != Dtype, "unsupported scalar conversion");
  }
}

ScalarValue ConvertScalar(ScalarValue value, uint64_t dtype) {
  if (dtype != kDtypePred && dtype <= kDtypeU64 && IntegralScalarDtype(dtype)) {
    const auto bits = ScalarIntegerBits(value, dtype);
    return {SignedScalarDtype(dtype) ? static_cast<long double>(BitCast<int64_t>(bits))
                                    : static_cast<long double>(bits), 0, true};
  }
  const auto convert = [](ScalarValue input, auto target) {
    constexpr auto target_dtype = decltype(target)::value;
    const auto converted = ConvertScalar<target_dtype>(input);
    return ReadScalar(&converted, DescriptorDtypeCode<target_dtype>());
  };
  switch (dtype) {
    case kDtypePred:
      return convert(value, std::integral_constant<ffi::DataType, ffi::PRED>{});
    case kDtypeF16:
      return convert(value, std::integral_constant<ffi::DataType, ffi::F16>{});
    case kDtypeBF16:
      return convert(value, std::integral_constant<ffi::DataType, ffi::BF16>{});
    case kDtypeF32:
      return convert(value, std::integral_constant<ffi::DataType, ffi::F32>{});
    case kDtypeF64:
      return convert(value, std::integral_constant<ffi::DataType, ffi::F64>{});
    case kDtypeC64:
      return convert(value, std::integral_constant<ffi::DataType, ffi::C64>{});
    case kDtypeC128:
      return convert(value, std::integral_constant<ffi::DataType, ffi::C128>{});
    default:
      throw std::invalid_argument("unsupported scalar conversion");
  }
}

ScalarValue MultiplyScalars(ScalarValue left, ScalarValue right, uint64_t dtype) {
  left = ConvertScalar(left, dtype);
  right = ConvertScalar(right, dtype);
  if (IntegralScalarDtype(dtype)) {
    const uint64_t bits = ScalarIntegerBits(left) * ScalarIntegerBits(right);
    return ConvertScalar({static_cast<long double>(bits), 0, true}, dtype);
  }
  if (dtype == kDtypeC64) {
    const auto result = FullComplexMultiply(
        static_cast<float>(left.real), static_cast<float>(left.imaginary),
        static_cast<float>(right.real), static_cast<float>(right.imaginary));
    return {result.real(), result.imag(), false};
  }
  if (dtype == kDtypeC128) {
    const auto result = FullComplexMultiply(
        static_cast<double>(left.real), static_cast<double>(left.imaginary),
        static_cast<double>(right.real), static_cast<double>(right.imaginary));
    return {result.real(), result.imag(), false};
  }
  const long double product = dtype == kDtypeF64
      ? static_cast<long double>(static_cast<double>(left.real) * static_cast<double>(right.real))
      : static_cast<long double>(static_cast<float>(left.real) * static_cast<float>(right.real));
  return ConvertScalar({product, 0, false}, dtype);
}

ScalarValue AddScalars(ScalarValue left, ScalarValue right, uint64_t dtype) {
  left = ConvertScalar(left, dtype);
  right = ConvertScalar(right, dtype);
  if (dtype == kDtypePred) {
    return {left.real != 0 || right.real != 0 ? 1.0L : 0.0L, 0, true};
  }
  if (IntegralScalarDtype(dtype)) {
    const uint64_t bits = ScalarIntegerBits(left) + ScalarIntegerBits(right);
    return ConvertScalar({static_cast<long double>(bits), 0, true}, dtype);
  }
  if (dtype == kDtypeF64 || dtype == kDtypeC128) {
    return ConvertScalar({static_cast<double>(left.real) + static_cast<double>(right.real),
                          static_cast<double>(left.imaginary) + static_cast<double>(right.imaginary), false}, dtype);
  }
  return ConvertScalar({static_cast<float>(left.real) + static_cast<float>(right.real),
                        static_cast<float>(left.imaginary) + static_cast<float>(right.imaginary), false}, dtype);
}

struct AddF32 {
  float operator()(float left, float right) const { return left + right; }
};

struct AddF16 {
  uint16_t operator()(uint16_t left, uint16_t right) const {
    return FloatToHalf(HalfToFloat(left) + HalfToFloat(right));
  }
};

struct AddBF16 {
  uint16_t operator()(uint16_t left, uint16_t right) const {
    return FloatToBFloat16(BFloat16ToFloat(left) + BFloat16ToFloat(right));
  }
};

struct AddC64 {
  std::complex<float> operator()(std::complex<float> left,
                                 std::complex<float> right) const {
    return {left.real() + right.real(), left.imag() + right.imag()};
  }
};

struct AddPred {
  bool operator()(bool left, bool right) const { return left || right; }
};

template <typename Element>
struct AddIntegerOp {
  Element operator()(Element left, Element right) const {
    return AddInteger(left, right);
  }
};

struct AddF64 {
  double operator()(double left, double right) const { return left + right; }
};

struct AddC128 {
  std::complex<double> operator()(std::complex<double> left,
                                  std::complex<double> right) const {
    return {left.real() + right.real(), left.imag() + right.imag()};
  }
};

struct DynamicScaleF32 {
  float operator()(float factor, float value) const { return factor * value; }
};

struct DynamicScaleF16 {
  uint16_t operator()(uint16_t factor, uint16_t value) const {
    return FloatToHalf(HalfToFloat(factor) * HalfToFloat(value));
  }
};

struct DynamicScaleBF16 {
  uint16_t operator()(uint16_t factor, uint16_t value) const {
    return FloatToBFloat16(BFloat16ToFloat(factor) *
                           BFloat16ToFloat(value));
  }
};

struct DynamicScaleC64 {
  std::complex<float> operator()(std::complex<float> factor,
                                 std::complex<float> value) const {
    return FullComplexMultiply(factor.real(), factor.imag(), value.real(),
                               value.imag());
  }
};

struct DynamicScalePred {
  bool operator()(bool factor, bool value) const { return factor && value; }
};

template <typename Element>
struct DynamicScaleInteger {
  Element operator()(Element factor, Element value) const {
    using Unsigned = std::make_unsigned_t<Element>;
    const Unsigned product = static_cast<Unsigned>(
        static_cast<uint64_t>(BitCast<Unsigned>(factor)) *
        static_cast<uint64_t>(BitCast<Unsigned>(value)));
    return BitCast<Element>(product);
  }
};

struct DynamicScaleF64 {
  double operator()(double factor, double value) const {
    return factor * value;
  }
};

struct DynamicScaleC128 {
  std::complex<double> operator()(std::complex<double> factor,
                                  std::complex<double> value) const {
    return FullComplexMultiply(factor.real(), factor.imag(), value.real(),
                               value.imag());
  }
};


template <ffi::DataType Source, ffi::DataType Result>
struct ScalarCast {
  auto operator()(ffi::NativeType<Source> value) const {
    return ConvertScalar<Result, Source>(value);
  }
};

template <typename RecordType>
ScalarValue ReadMappingFactor(const RecordType& record) {
  const uint64_t bits[2] = {record.scale_real_bits, record.scale_imaginary_bits};
  if (record.mapped_dtype == kDtypeC64) {
    const std::complex<float> value{
        BitCast<float>(static_cast<uint32_t>(bits[0])),
        BitCast<float>(static_cast<uint32_t>(bits[1]))};
    return ReadScalar(&value, kDtypeC64);
  }
  return ReadScalar(bits, record.mapped_dtype);
}

struct BoundMapping {
  ScalarValue factor;
  uint64_t dtype;
  bool identity;

  template <typename RecordType>
  static BoundMapping Bind(const RecordType& record) {
    return {record.identity_scale ? ScalarValue{} : ReadMappingFactor(record),
            record.mapped_dtype, record.identity_scale};
  }

  ScalarValue operator()(ScalarValue source) const {
    return identity ? source : MultiplyScalars(factor, source, dtype);
  }
};

template <ffi::DataType Source, ffi::DataType Result>
struct TypedScalarMapping {
  BoundMapping mapping;
  auto operator()(ffi::NativeType<Source> value) const {
    return ConvertScalar<Result>(mapping(ReadScalar(&value, DescriptorDtypeCode<Source>())));
  }
};

template <ffi::DataType Dtype, typename Scale, typename Map = IdentityScalar>
struct StaticScale {
  template <typename RecordType>
  auto Bind(const RecordType& record) const {
    auto factor = ConvertScalar<Dtype>(ReadMappingFactor(record));
    return BoundScalarScale<decltype(factor), Scale, Map>{factor, Scale{}, Map{}};
  }
};

struct NarrowF16Product {
  float factor;
  uint16_t operator()(float value) const { return FloatToHalf(factor * value); }
  template <typename RecordType>
  NarrowF16Product Bind(const RecordType& record) const {
    return {BitCast<float>(static_cast<uint32_t>(record.scale_real_bits))};
  }
};

template <typename Real>
struct ProjectedComplexProduct {
  std::complex<Real> factor;

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((noinline))
#endif
  Real operator()(std::complex<Real> value) const {
    return FullComplexProductReal(factor.real(), factor.imag(), value.real(), value.imag());
  }

  template <typename RecordType>
  ProjectedComplexProduct Bind(const RecordType& record) const {
    const auto value = ReadMappingFactor(record);
    return {{static_cast<Real>(value.real), static_cast<Real>(value.imaginary)}};
  }
};

template <typename Real>
struct EmbeddedRealProduct {
  Real factor;
  std::complex<Real> operator()(Real value) const { return {factor * value, Real{0}}; }
  template <typename RecordType>
  EmbeddedRealProduct Bind(const RecordType& record) const {
    return {static_cast<Real>(ReadMappingFactor(record).real)};
  }
};

template <typename Result>
struct NarrowIntegerProduct {
  Result factor;
  Result operator()(int32_t value) const {
    using Unsigned = std::make_unsigned_t<Result>;
    const uint32_t product = BitCast<uint32_t>(value) *
                            BitCast<uint32_t>(static_cast<int32_t>(factor));
    return BitCast<Result>(static_cast<Unsigned>(product));
  }
  template <typename RecordType>
  NarrowIntegerProduct Bind(const RecordType& record) const {
    return {BitCast<Result>(static_cast<std::make_unsigned_t<Result>>(record.scale_real_bits))};
  }
};

struct NarrowUnsignedProduct {
  uint8_t factor;
  uint8_t operator()(uint32_t value) const { return static_cast<uint8_t>(value * factor); }
  template <typename RecordType>
  NarrowUnsignedProduct Bind(const RecordType& record) const {
    return {static_cast<uint8_t>(record.scale_real_bits)};
  }
};

using F16Scale = BoundScalarScale<ffi::NativeType<ffi::F16>, DynamicScaleF16>;
using F16Mapping = StaticScale<ffi::F16, DynamicScaleF16>;
using BF16Mapping = StaticScale<ffi::BF16, DynamicScaleBF16>;
using F32Scale = BoundScalarScale<ffi::NativeType<ffi::F32>, DynamicScaleF32>;
using F32Mapping = StaticScale<ffi::F32, DynamicScaleF32>;
using F64Mapping = StaticScale<ffi::F64, DynamicScaleF64>;
using C64Scale = BoundScalarScale<ffi::NativeType<ffi::C64>, DynamicScaleC64>;
using C64Mapping = StaticScale<ffi::C64, DynamicScaleC64>;
using C128Mapping = StaticScale<ffi::C128, DynamicScaleC128>;
using PREDMapping = StaticScale<ffi::PRED, DynamicScalePred>;
template <typename Element>
using IntegerMapping = StaticScale<
    std::is_signed_v<Element> ? (sizeof(Element) == 1 ? ffi::S8 : sizeof(Element) == 2 ? ffi::S16 : sizeof(Element) == 4 ? ffi::S32 : ffi::S64)
                             : (sizeof(Element) == 1 ? ffi::U8 : sizeof(Element) == 2 ? ffi::U16 : sizeof(Element) == 4 ? ffi::U32 : ffi::U64),
    DynamicScaleInteger<Element>>;
using F16F32Scale = BoundScalarScale<ffi::NativeType<ffi::F32>, DynamicScaleF32, ScalarCast<ffi::F16, ffi::F32>>;
using F16F32Mapping = StaticScale<ffi::F32, DynamicScaleF32, ScalarCast<ffi::F16, ffi::F32>>;
using F32C64Scale = BoundScalarScale<ffi::NativeType<ffi::C64>, DynamicScaleC64, ScalarCast<ffi::F32, ffi::C64>>;
using F32C64Mapping = StaticScale<ffi::C64, DynamicScaleC64, ScalarCast<ffi::F32, ffi::C64>>;
using C64F32Scale = BoundScalarScale<ffi::NativeType<ffi::F32>, DynamicScaleF32, ScalarCast<ffi::C64, ffi::F32>>;
using C64F32Mapping = StaticScale<ffi::F32, DynamicScaleF32, ScalarCast<ffi::C64, ffi::F32>>;
using F64C128Mapping = StaticScale<ffi::C128, DynamicScaleC128, ScalarCast<ffi::F64, ffi::C128>>;
using C128F64Mapping = StaticScale<ffi::F64, DynamicScaleF64, ScalarCast<ffi::C128, ffi::F64>>;
using S32PREDMapping = StaticScale<ffi::PRED, DynamicScalePred, ScalarCast<ffi::S32, ffi::PRED>>;

int ScalarCoefficientKind(ScalarValue value) {
  if (value.real == 0 && value.imaginary == 0) {
    return 0;
  }
  return value.real == 1 && value.imaginary == 0 ? 1 : 2;
}

template <ffi::DataType Dtype>
struct PromotedScaleF64 {
  double factor;
  auto operator()(ffi::NativeType<Dtype> value) const {
    if constexpr (Dtype == ffi::F32) {
      return static_cast<float>(factor * static_cast<double>(value));
    } else {
      return ScalarCast<ffi::F64, Dtype>{}(factor * ScalarCast<Dtype, ffi::F64>{}(value));
    }
  }
};

template <ffi::DataType Dtype>
struct TypedScaleOperation {
  ScalarValue coefficient;
  uint64_t operation_dtype;

  auto operator()(ffi::NativeType<Dtype> value) const {
    const auto input = ReadScalar(&value, DescriptorDtypeCode<Dtype>());
    return ConvertScalar<Dtype>(MultiplyScalars(coefficient, input, operation_dtype));
  }
};

template <typename Execute>
bool BindNativeArithmetic(uint64_t dtype, Execute execute) {
  switch (dtype) {
    case kDtypeF16:
      return execute(std::integral_constant<ffi::DataType, ffi::F16>{}, DynamicScaleF16{}, AddF16{});
    case kDtypeBF16:
      return execute(std::integral_constant<ffi::DataType, ffi::BF16>{}, DynamicScaleBF16{}, AddBF16{});
    case kDtypeF32:
      return execute(std::integral_constant<ffi::DataType, ffi::F32>{}, DynamicScaleF32{}, AddF32{});
    case kDtypeF64:
      return execute(std::integral_constant<ffi::DataType, ffi::F64>{}, DynamicScaleF64{}, AddF64{});
    case kDtypeC64:
      return execute(std::integral_constant<ffi::DataType, ffi::C64>{}, DynamicScaleC64{}, AddC64{});
    case kDtypeC128:
      return execute(std::integral_constant<ffi::DataType, ffi::C128>{}, DynamicScaleC128{}, AddC128{});
    default:
      return false;
  }
}

template <ffi::DataType Source, ffi::DataType Result, ffi::DataType Compute, typename Scale>
struct NativeScaleOperation {
  ffi::NativeType<Compute> factor;
  Scale scale;

  auto operator()(ffi::NativeType<Source> value) const {
    return ConvertScalar<Result, Compute>(scale(factor, ConvertScalar<Compute, Source>(value)));
  }
};

template <ffi::DataType Source, ffi::DataType Result, typename Execute>
bool BindNativeScale(ScalarValue factor, uint64_t dtype, Execute execute) {
  if constexpr (kFloatingScalarDtype<Source> && kFloatingScalarDtype<Result>) {
    return BindNativeArithmetic(dtype, [&](auto type, auto scale, auto) {
      constexpr auto compute = decltype(type)::value;
      if constexpr (compute == Result ||
                    (Source == Result && compute == ffi::F64 &&
                     (Result == ffi::F16 || Result == ffi::BF16 || Result == ffi::F32))) {
        return false;
      } else {
        execute(NativeScaleOperation<Source, Result, compute, decltype(scale)>{
            ConvertScalar<compute>(factor), scale});
        return true;
      }
    });
  }
  return false;
}

template <ffi::DataType Dtype, bool Differential, typename Scale, typename Execute>
void BindScaleOperation(ScalarValue coefficient, uint64_t operation_dtype,
                        int coefficient_kind, Scale scale, Execute execute) {
  if constexpr (!Differential) {
    if (coefficient_kind == 0) {
      execute(ZeroScalar<ffi::NativeType<Dtype>>{});
      return;
    }
    if (coefficient_kind == 1) {
      execute(IdentityScalar{});
      return;
    }
  }
  if constexpr (Dtype == ffi::F16 || Dtype == ffi::BF16 || Dtype == ffi::F32) {
    if (operation_dtype == kDtypeF64) {
      execute(PromotedScaleF64<Dtype>{static_cast<double>(coefficient.real)});
      return;
    }
  }
  if (operation_dtype != DescriptorDtypeCode<Dtype>()) {
    if (BindNativeScale<Dtype, Dtype>(coefficient, operation_dtype, execute)) return;
    execute(TypedScaleOperation<Dtype>{coefficient, operation_dtype});
    return;
  }
  auto factor = ConvertScalar<Dtype>(coefficient);
  execute(BoundScalarScale<decltype(factor), Scale>{factor, scale});
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename RecordType, typename Map, typename Execute>
void BindScalarMapping(const RecordType& record, uint64_t expected_dtype,
                       Map map, Execute execute) {
  if (record.identity_scale) {
    if constexpr (SourceDtype == ResultDtype) execute(IdentityScalar{});
    else execute(ScalarCast<SourceDtype, ResultDtype>{});
  } else if (record.mapped_dtype != expected_dtype) {
    execute(TypedScalarMapping<SourceDtype, ResultDtype>{BoundMapping::Bind(record)});
  } else {
    execute(map.Bind(record));
  }
}

template <ffi::DataType SourceDtype>
struct TypedUpdateTerm {
  ScalarValue factor;
  uint64_t product_dtype;
  int source_kind;

  ScalarValue operator()(ffi::NativeType<SourceDtype> value) const {
    if (source_kind == 0) return ScalarValue{};
    const auto input = ReadScalar(&value, DescriptorDtypeCode<SourceDtype>());
    return source_kind == 1 ? input
                            : MultiplyScalars(factor, input, product_dtype);
  }
};

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype>
struct TypedUpdateSource {
  ScalarValue factor;
  uint64_t product_dtype;

  auto operator()(ffi::NativeType<SourceDtype> value) const {
    const auto mapped = MultiplyScalars(
        factor, ReadScalar(&value, DescriptorDtypeCode<SourceDtype>()), product_dtype);
    return ConvertScalar<ResultDtype>(mapped);
  }
};

template <ffi::DataType ResultDtype>
struct CombineTypedOriginalValue {
  const ffi::NativeType<ResultDtype>* base;
  ffi::NativeType<ResultDtype>* output;
  ScalarValue factor;
  uint64_t product_dtype;
  uint64_t addition_dtype;
  int source_kind;
  int base_kind;

  void operator()(ffi::NativeType<ResultDtype>& destination, ScalarValue mapped) const {
    ScalarValue previous;
    if (base_kind != 0) {
      const auto input = ReadScalar(base + (&destination - output),
                                     DescriptorDtypeCode<ResultDtype>());
      previous = base_kind == 1 ? input
                               : MultiplyScalars(factor, input, product_dtype);
    }
    const auto combined = source_kind == 0 ? previous
        : base_kind == 0 ? mapped
        : AddScalars(mapped, previous, addition_dtype);
    destination = ConvertScalar<ResultDtype>(combined);
  }
};

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename Scale, typename Execute>
void BindUpdateSource(ScalarValue factor, int source_kind,
                      uint64_t product_dtype,
                      Scale scale, Execute execute) {
  const auto result_dtype = DescriptorDtypeCode<ResultDtype>();
  if (source_kind == 0) {
    execute(ZeroScalar<ffi::NativeType<ResultDtype>>{});
    return;
  }
  if (source_kind == 1) {
    if constexpr (SourceDtype == ResultDtype) execute(IdentityScalar{});
    else execute(ScalarCast<SourceDtype, ResultDtype>{});
    return;
  }
  if constexpr (SourceDtype == ResultDtype) {
    BindScaleOperation<ResultDtype, false>(factor, product_dtype, source_kind, scale, execute);
    return;
  }
  if (product_dtype == result_dtype) {
    auto coefficient = ConvertScalar<ResultDtype>(factor);
    execute(BoundScalarScale<decltype(coefficient), Scale,
        ScalarCast<SourceDtype, ResultDtype>>{coefficient, scale, {}});
    return;
  }
  if (BindNativeScale<SourceDtype, ResultDtype>(factor, product_dtype, execute)) return;
  execute(TypedUpdateSource<SourceDtype, ResultDtype>{
      factor, product_dtype});
}

template <bool HasSource, int BaseKind, typename Output, typename Value,
          typename Scale, typename Add, typename BaseMap = IdentityScalar,
          typename ResultMap = IdentityScalar>
struct CombineOriginalValue {
  const Output* base;
  Output* output;
  Value factor;
  Scale scale;
  Add add;
  BaseMap base_map{};
  ResultMap result_map{};

  template <typename Mapped>
  void operator()(Output& destination, Mapped mapped) const {
    const auto original = base_map(base[&destination - output]);
    const auto previous = [&] {
      if constexpr (BaseKind == 1) return original;
      else return scale(factor, original);
    }();
    if constexpr (HasSource) destination = result_map(add(mapped, previous));
    else destination = result_map(previous);
  }
};


template <typename Output, typename Value, typename Scale, typename Add,
          typename Map, typename Execute>
void BindUpdateCombination(const Output* base, Output* output,
                           Value source_factor, Value base_factor,
                           int source_kind, int base_kind, Scale scale,
                           Add add, Map mapping, Execute execute) {
  const auto bind = [&](auto source_mode, auto base_mode) {
    const auto write_op =
        CombineOriginalValue<decltype(source_mode)::value != 0,
            decltype(base_mode)::value, Output, Value, Scale, Add>{
                base, output, base_factor, scale, add};
    if constexpr (decltype(source_mode)::value == 0) {
      execute(ZeroScalar<Output>{}, write_op);
    } else if constexpr (decltype(source_mode)::value == 1) {
      execute(mapping, write_op);
    } else {
      execute(BoundScalarScale<Value, Scale, Map>{source_factor, scale, mapping},
              write_op);
    }
  };
  const auto with_source = [&](auto source_mode) {
    if (base_kind == 1) {
      bind(source_mode, std::integral_constant<int, 1>{});
    } else {
      bind(source_mode, std::integral_constant<int, 2>{});
    }
  };
  switch (source_kind) {
    case 0: with_source(std::integral_constant<int, 0>{}); break;
    case 1: with_source(std::integral_constant<int, 1>{}); break;
    default: with_source(std::integral_constant<int, 2>{}); break;
  }
}

template <ffi::DataType Source, ffi::DataType Result, typename Execute>
bool BindNativeUpdateCombination(const ffi::NativeType<Result>* base,
                                 ffi::NativeType<Result>* output,
                                 ScalarValue source_factor, ScalarValue base_factor,
                                 int source_kind, int base_kind,
                                 uint64_t source_dtype, uint64_t base_dtype,
                                 uint64_t dtype, Execute execute) {
  if constexpr (kFloatingScalarDtype<Source> && kFloatingScalarDtype<Result>) {
    if constexpr (Source == ffi::F32 && Result == ffi::F32) {
      if (source_kind != 0 && dtype == kDtypeF64 &&
          ((source_dtype == kDtypeF32 && base_dtype == kDtypeF64) ||
           (source_dtype == kDtypeF64 && base_dtype == kDtypeF32))) {
        const auto bind = [&](auto source_wide, auto base_mode) {
          constexpr auto source_compute = decltype(source_wide)::value ? ffi::F64 : ffi::F32;
          constexpr auto base_compute = decltype(source_wide)::value ? ffi::F32 : ffi::F64;
          using SourceScale = std::conditional_t<decltype(source_wide)::value, DynamicScaleF64, DynamicScaleF32>;
          using BaseScale = std::conditional_t<decltype(source_wide)::value, DynamicScaleF32, DynamicScaleF64>;
          using SourceValue = ffi::NativeType<source_compute>;
          using BaseValue = ffi::NativeType<base_compute>;
          const auto write_op = CombineOriginalValue<true, decltype(base_mode)::value,
              float, BaseValue, BaseScale, AddF64, ScalarCast<ffi::F32, base_compute>,
              ScalarCast<ffi::F64, ffi::F32>>{
                  base, output, ConvertScalar<base_compute>(base_factor), {}, {}};
          if constexpr (source_compute == ffi::F32) {
            if (source_kind == 1) {
              execute(IdentityScalar{}, write_op);
              return;
            }
          }
          execute(BoundScalarScale<SourceValue, SourceScale, ScalarCast<ffi::F32, source_compute>>{
              ConvertScalar<source_compute>(source_factor), {}, {}}, write_op);
        };
        if (source_dtype == kDtypeF32) {
          bind(std::false_type{}, std::integral_constant<int, 2>{});
        } else if (base_kind == 1) {
          bind(std::true_type{}, std::integral_constant<int, 1>{});
        } else {
          bind(std::true_type{}, std::integral_constant<int, 2>{});
        }
        return true;
      }
    }
    if ((source_kind != 0 && source_dtype != dtype) || base_dtype != dtype) return false;
    return BindNativeArithmetic(dtype, [&](auto type, auto scale, auto add) {
      constexpr auto compute = decltype(type)::value;
      if constexpr (compute == Result) {
        return false;
      } else {
        using Storage = ffi::NativeType<Result>;
        using Value = ffi::NativeType<compute>;
        using BaseMap = ScalarCast<Result, compute>;
        using ResultMap = ScalarCast<compute, Result>;
        const auto second = ConvertScalar<compute>(base_factor);
        if (source_kind == 0) {
          execute(ZeroScalar<Value>{},
              CombineOriginalValue<false, 2, Storage, Value, decltype(scale), decltype(add),
                  BaseMap, ResultMap>{base, output, second, scale, add});
          return true;
        }
        const auto write_op =
            CombineOriginalValue<true, 2, Storage, Value, decltype(scale), decltype(add),
                BaseMap, ResultMap>{base, output, second, scale, add};
        if constexpr (Source == compute) {
          if (source_kind == 1) {
            execute(IdentityScalar{}, write_op);
            return true;
          }
        }
        execute(BoundScalarScale<Value, decltype(scale), ScalarCast<Source, compute>>{
            ConvertScalar<compute>(source_factor), scale, {}}, write_op);
        return true;
      }
    });
  }
  return false;
}
