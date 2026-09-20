#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "numeric/scalar.h"
#include "ffi/dtype.h"

using namespace tensor0::stride;
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;




template <ffi::DataType Dtype>
void CheckBuffer() {
  using Scalar = ScalarDtype<Dtype>;
  auto value = scalar::Convert<Scalar, scalar::S32>(2);
  XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      static_cast<XLA_FFI_DataType>(Dtype), &value, 0, nullptr};
  unsigned visits = 0;
  VisitCoefficient(ffi::AnyBuffer(&buffer), [&](auto type, const auto* data) {
    ++visits;
    if constexpr (std::is_same_v<Scalar, typename decltype(type)::type>) {
      assert(data == &value && *data == value);
    } else {
      assert(false);
    }
  });
  assert(visits == 1);
}

void CheckRemainingArgs() {
  float source = 3;
  int64_t source_size = 1;
  int32_t integer_factor = 2;
  float real_factor = 0.5f;
  std::complex<float> complex_factor{0, 1};
  std::array<XLA_FFI_Buffer, 4> buffers{{
      {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32, &source, 1, &source_size},
      {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_S32, &integer_factor, 0, nullptr},
      {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32, &real_factor, 0, nullptr},
      {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_C64, &complex_factor, 0, nullptr}}};
  std::array<XLA_FFI_ArgType, 4> kinds;
  kinds.fill(XLA_FFI_ArgType_BUFFER);
  std::array<void*, 4> pointers{&buffers[0], &buffers[1], &buffers[2], &buffers[3]};
  const XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, 4, kinds.data(), pointers.data()};
  const ffi::RemainingArgs coefficients(&args, 1);
  assert(coefficients.size() == 3);
  for (int repeat = 0; repeat < 2; ++repeat) {
    const std::array<std::complex<float>, 3> expected{
        std::complex<float>{static_cast<float>(integer_factor), 0},
        std::complex<float>{real_factor, 0}, complex_factor};
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
      const auto buffer = coefficients.get<ffi::AnyBuffer>(index);
      assert(buffer.has_value());
      assert(buffer->dimensions().size() == 0);
      assert(buffer->element_type() == static_cast<ffi::DataType>(buffers[index + 1].dtype));
      VisitCoefficient(*buffer, [&](auto type, const auto* data) {
        using Dtype = typename decltype(type)::type;
        assert((scalar::Convert<scalar::C64, Dtype>(*data) == expected[index]));
      });
    }
    integer_factor = -3;
    real_factor = 1.5f;
    complex_factor = {2, -1};
  }
  assert(!coefficients.get<ffi::AnyBuffer>(3).has_value());
  assert(ffi::RemainingArgs(&args, 4).empty());
}

void CheckUnsupported() {
  for (auto dtype : {ffi::DataType::INVALID, ffi::TOKEN, ffi::S4, ffi::F8E5M2}) {
    XLA_FFI_Buffer buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
        static_cast<XLA_FFI_DataType>(dtype), nullptr, 0, nullptr};
    bool rejected = false;
    try {
      VisitCoefficient(ffi::AnyBuffer(&buffer), [](auto, const auto*) { assert(false); });
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected);
  }
}

int main() {
  CheckBuffer<ffi::PRED>();
  CheckBuffer<ffi::S8>();
  CheckBuffer<ffi::S16>();
  CheckBuffer<ffi::S32>();
  CheckBuffer<ffi::S64>();
  CheckBuffer<ffi::U8>();
  CheckBuffer<ffi::U16>();
  CheckBuffer<ffi::U32>();
  CheckBuffer<ffi::U64>();
  CheckBuffer<ffi::F16>();
  CheckBuffer<ffi::BF16>();
  CheckBuffer<ffi::F32>();
  CheckBuffer<ffi::F64>();
  CheckBuffer<ffi::C64>();
  CheckBuffer<ffi::C128>();
  CheckRemainingArgs();
  CheckUnsupported();
}
