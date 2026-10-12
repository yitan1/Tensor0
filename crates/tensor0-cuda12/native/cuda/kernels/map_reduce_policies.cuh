#pragma once

#include "../arithmetic.cuh"

#include <type_traits>

namespace tensor0::stride::cuda {

// Storage is copied without arithmetic conversions, preserving all payload bits.
template <int Bytes> struct Storage { unsigned char bytes[Bytes]; };

template <int Bytes> struct CopyMap {
  using Value = Storage<Bytes>;
  const Value* source;
  Value* output;
  uint64_t source_size, output_size;

  __device__ uint64_t Begin(uint64_t batch) const { return batch; }
  __device__ bool Active(uint64_t) const { return true; }
  __device__ Value Initial(uint64_t, uint64_t, int64_t) const { return {}; }
  __device__ Value Term(uint64_t, uint64_t batch, int64_t first, int64_t) const {
    return source[batch * source_size + first];
  }
  __device__ Value Combine(Value, Value term) const { return term; }
  __device__ void Store(Value value, uint64_t batch, int64_t second) const {
    output[batch * output_size + second] = value;
  }
};

template <typename T, typename Real, typename Alpha, typename Beta> struct UpdateMap {
  const T* source;
  const T* base;
  T* output;
  const Alpha* alpha;
  uint64_t alpha_count;
  const Beta* beta;
  uint64_t beta_count;
  uint64_t source_size, output_size;

  struct Factors {
    std::conditional_t<std::is_same_v<Alpha, int32_t>, Real, Alpha> a;
    std::conditional_t<std::is_same_v<Beta, int32_t>, Real, Beta> b;
  };
  __device__ Factors Begin(uint64_t batch) const {
    return {arithmetic::ReadCoefficient<Real>(alpha, alpha_count, batch),
            arithmetic::ReadCoefficient<Real>(beta, beta_count, batch)};
  }
  __device__ bool Active(Factors) const { return true; }
  __device__ T Initial(Factors, uint64_t, int64_t) const { return {}; }
  __device__ T Term(Factors factors, uint64_t batch, int64_t first, int64_t second) const {
    const auto src = batch * source_size + first;
    const auto dst = batch * output_size + second;
    const auto a = factors.a;
    const auto b = factors.b;
    if (arithmetic::Zero(a)) {
      if (arithmetic::Zero(b)) return T{};
      return arithmetic::One(b) ? base[dst] : arithmetic::Product(b, base[dst]);
    }
    if (arithmetic::Zero(b))
      return arithmetic::One(a) ? source[src] : arithmetic::Product(a, source[src]);
    const auto source_term = arithmetic::One(a) ? source[src] : arithmetic::Product(a, source[src]);
    const auto base_term = arithmetic::One(b) ? base[dst] : arithmetic::Product(b, base[dst]);
    return arithmetic::Sum(source_term, base_term);
  }
  __device__ T Combine(T, T term) const { return term; }
  __device__ void Store(T value, uint64_t batch, int64_t second) const {
    output[batch * output_size + second] = value;
  }
};

template <typename T, typename Real, typename Raw> struct SumFiber {
  using Value = T;
  const T* source;
  T* output;
  const Raw* coefficient;
  uint64_t coefficient_count, source_size, output_size;

  __device__ auto Begin(uint64_t batch) const {
    return coefficient == nullptr ? Raw{1} :
        arithmetic::ReadCoefficient<Real>(coefficient, coefficient_count, batch);
  }
  template <typename Factor> __device__ bool Active(Factor factor) const {
    return !arithmetic::Zero(factor);
  }
  template <typename Factor> __device__ T Initial(Factor, uint64_t batch, int64_t second) const {
    return output[batch * output_size + second];
  }
  template <typename Factor> __device__ T Term(Factor factor, uint64_t batch, int64_t first, int64_t) const {
    const T value = source[batch * source_size + first];
    return arithmetic::One(factor) ? value : arithmetic::Product(factor, value);
  }
  __device__ T Combine(T value, T term) const { return arithmetic::Sum(value, term); }
  __device__ void Store(T value, uint64_t batch, int64_t second) const {
    output[batch * output_size + second] = value;
  }
};

template <typename T> __device__ T ConjugateLeft(T value) { return value; }
template <typename Real> __device__ arithmetic::Complex<Real> ConjugateLeft(arithmetic::Complex<Real> value) {
  return {value.real, -value.imag};
}

template <typename T> struct DotFiber {
  using Value = T;
  const T* left;
  const T* right;
  T* output;
  uint64_t left_size, right_size;
  bool conjugate;

  __device__ uint64_t Begin(uint64_t batch) const { return batch; }
  __device__ bool Active(uint64_t) const { return true; }
  __device__ T Initial(uint64_t, uint64_t batch, int64_t) const { return output[batch]; }
  __device__ T Term(uint64_t, uint64_t batch, int64_t first, int64_t second) const {
    T a = left[batch * left_size + first];
    if (conjugate) a = ConjugateLeft(a);
    return arithmetic::Product(a, right[batch * right_size + second]);
  }
  __device__ T Combine(T value, T term) const { return arithmetic::Sum(value, term); }
  __device__ void Store(T value, uint64_t batch, int64_t) const { output[batch] = value; }
};

}  // namespace tensor0::stride::cuda
