#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace tensor0::stride::cuda::arithmetic {

namespace ffi = xla::ffi;

// Device storage has the same two-component representation as FFI complex values.
template <typename Real> struct Complex { Real real, imag; };

__device__ inline float Product(float a, float b) { return __fmul_rn(a, b); }
__device__ inline double Product(double a, double b) { return __dmul_rn(a, b); }
__device__ inline float Sum(float a, float b) { return __fadd_rn(a, b); }
__device__ inline double Sum(double a, double b) { return __dadd_rn(a, b); }
__device__ inline float Fused(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ inline double Fused(double a, double b, double c) { return __fma_rn(a, b, c); }

template <typename Real>
__device__ Complex<Real> Product(Real a, Complex<Real> b) {
  return {Product(a, b.real), Product(a, b.imag)};
}

template <typename Real>
__device__ Complex<Real> Product(Complex<Real> a, Complex<Real> b) {
  // Match the existing full-complex expression, without promoting real factors
  // to complex or contracting the separate real products/additions elsewhere.
  return {Fused(a.real, b.real, -Product(a.imag, b.imag)),
          Fused(a.imag, b.real, Product(a.real, b.imag))};
}

template <typename Real>
__device__ Complex<Real> Sum(Complex<Real> a, Complex<Real> b) {
  return {Sum(a.real, b.real), Sum(a.imag, b.imag)};
}

template <typename T> __device__ bool Zero(T value) { return value == T(0); }
template <typename T> __device__ bool One(T value) { return value == T(1); }
template <typename Real> __device__ bool Zero(Complex<Real> value) {
  return value.real == Real(0) && value.imag == Real(0);
}
template <typename Real> __device__ bool One(Complex<Real> value) {
  return value.real == Real(1) && value.imag == Real(0);
}

template <typename Real, typename Raw>
__device__ auto ReadCoefficient(const Raw* data, uint64_t count, uint64_t batch) {
  const auto value = data[count == 1 ? 0 : batch];
  if constexpr (std::is_same_v<Raw, int32_t>) return static_cast<Real>(value);
  else return value;
}

template <typename T, typename Real, typename Function>
void VisitCoefficient(ffi::DataType dtype, Function function) {
  constexpr auto real_dtype = std::is_same_v<Real, float> ? ffi::F32 : ffi::F64;
  if (dtype == ffi::S32) { function(std::type_identity<int32_t>{}); return; }
  if (dtype == real_dtype) { function(std::type_identity<Real>{}); return; }
  if constexpr (!std::is_same_v<T, Real>) {
    constexpr auto complex_dtype = std::is_same_v<Real, float> ? ffi::C64 : ffi::C128;
    if (dtype == complex_dtype) { function(std::type_identity<T>{}); return; }
  }
  throw std::invalid_argument("unsupported CUDA coefficient dtype; expected S32 or matching real/complex precision");
}

}  // namespace tensor0::stride::cuda::arithmetic
