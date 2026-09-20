#pragma once

#include "../numeric/scalar.h"
#include "xla/ffi/api/ffi.h"
#include <stdexcept>
#include <type_traits>

namespace tensor0::stride {

namespace ffi = xla::ffi;

template <ffi::DataType Dtype>
struct ScalarDtypeMapping;

template <> struct ScalarDtypeMapping<ffi::PRED> { using Type = scalar::Pred; };
template <> struct ScalarDtypeMapping<ffi::S8> { using Type = scalar::S8; };
template <> struct ScalarDtypeMapping<ffi::S16> { using Type = scalar::S16; };
template <> struct ScalarDtypeMapping<ffi::S32> { using Type = scalar::S32; };
template <> struct ScalarDtypeMapping<ffi::S64> { using Type = scalar::S64; };
template <> struct ScalarDtypeMapping<ffi::U8> { using Type = scalar::U8; };
template <> struct ScalarDtypeMapping<ffi::U16> { using Type = scalar::U16; };
template <> struct ScalarDtypeMapping<ffi::U32> { using Type = scalar::U32; };
template <> struct ScalarDtypeMapping<ffi::U64> { using Type = scalar::U64; };
template <> struct ScalarDtypeMapping<ffi::F16> { using Type = scalar::F16; };
template <> struct ScalarDtypeMapping<ffi::BF16> { using Type = scalar::BF16; };
template <> struct ScalarDtypeMapping<ffi::F32> { using Type = scalar::F32; };
template <> struct ScalarDtypeMapping<ffi::F64> { using Type = scalar::F64; };
template <> struct ScalarDtypeMapping<ffi::C64> { using Type = scalar::C64; };
template <> struct ScalarDtypeMapping<ffi::C128> { using Type = scalar::C128; };

template <ffi::DataType Dtype>
using ScalarDtype = typename ScalarDtypeMapping<Dtype>::Type;

template <typename Function>
void VisitScalarDtype(ffi::DataType dtype, Function function) {
  switch (dtype) {
    case ffi::PRED: function(std::type_identity<ScalarDtype<ffi::PRED>>{}); return;
    case ffi::S8: function(std::type_identity<ScalarDtype<ffi::S8>>{}); return;
    case ffi::S16: function(std::type_identity<ScalarDtype<ffi::S16>>{}); return;
    case ffi::S32: function(std::type_identity<ScalarDtype<ffi::S32>>{}); return;
    case ffi::S64: function(std::type_identity<ScalarDtype<ffi::S64>>{}); return;
    case ffi::U8: function(std::type_identity<ScalarDtype<ffi::U8>>{}); return;
    case ffi::U16: function(std::type_identity<ScalarDtype<ffi::U16>>{}); return;
    case ffi::U32: function(std::type_identity<ScalarDtype<ffi::U32>>{}); return;
    case ffi::U64: function(std::type_identity<ScalarDtype<ffi::U64>>{}); return;
    case ffi::F16: function(std::type_identity<ScalarDtype<ffi::F16>>{}); return;
    case ffi::BF16: function(std::type_identity<ScalarDtype<ffi::BF16>>{}); return;
    case ffi::F32: function(std::type_identity<ScalarDtype<ffi::F32>>{}); return;
    case ffi::F64: function(std::type_identity<ScalarDtype<ffi::F64>>{}); return;
    case ffi::C64: function(std::type_identity<ScalarDtype<ffi::C64>>{}); return;
    case ffi::C128: function(std::type_identity<ScalarDtype<ffi::C128>>{}); return;
    default: throw std::invalid_argument("unsupported scalar dtype");
  }
}

template <typename Function>
void VisitCoefficient(ffi::AnyBuffer buffer, Function function) {
  VisitScalarDtype(buffer.element_type(), [&](auto type) {
    using Dtype = typename decltype(type)::type;
    using Value = scalar::Value<Dtype>;
    if constexpr (scalar::kIsHalf<Dtype>) {
      function(type, buffer.reinterpret_data<Value>());
    } else {
      function(type, buffer.typed_data<Value>());
    }
  });
}

}

#define TENSOR0_STRIDE_FOR_EACH_DTYPE(Define) \
  Define(S32, S32) \
  Define(F32, F32) \
  Define(F16, F16) \
  Define(BF16, BF16) \
  Define(C64, C64) \
  Define(F64, F64) \
  Define(C128, C128) \
  Define(S64, S64) \
  Define(U64, U64) \
  Define(S16, S16) \
  Define(S8, S8) \
  Define(U8, U8) \
  Define(U16, U16) \
  Define(U32, U32) \
  Define(Pred, PRED)
