#pragma once

#include "../arithmetic.cuh"

namespace tensor0::stride::cuda::dot {
using namespace arithmetic;
constexpr uint64_t kChunk = 1024;
constexpr unsigned kThreads = 256;

template <typename T> __device__ T Conjugate(T value) { return value; }
template <typename R> __device__ Complex<R> Conjugate(Complex<R> value) {
  return {value.real, -value.imag};
}

template <typename T>
__device__ T Reduce(T value, T* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned width = kThreads / 2; width != 0; width /= 2) {
    if (threadIdx.x < width) shared[threadIdx.x] = Sum(shared[threadIdx.x], shared[threadIdx.x + width]);
    __syncthreads();
  }
  return shared[0];
}

template <typename T>
__global__ void Partials(const T* left, const T* right, T* scratch,
                         const int64_t* record, uint64_t count, uint64_t chunks,
                         uint64_t capacity, uint64_t total, uint64_t left_size,
                         uint64_t right_size, bool conjugate) {
  __shared__ T shared[kThreads];
  for (uint64_t task = blockIdx.x; task < total;) {
    const uint64_t batch = task / chunks, chunk = task % chunks;
    const uint64_t begin = chunk * kChunk;
    const uint64_t length = min(kChunk, count - begin);
    T value{};
    for (uint64_t local = threadIdx.x; local < length; local += kThreads) {
      uint64_t logical = begin + local;
      int64_t lhs = record[1], rhs = record[2];
      const int64_t rank = record[0];
      for (int64_t axis = rank; axis-- > 0;) {
        const auto coordinate = static_cast<int64_t>(logical % record[3 + axis]);
        logical /= record[3 + axis];
        lhs += coordinate * record[3 + rank + axis];
        rhs += coordinate * record[3 + 2 * rank + axis];
      }
      const T a = left[batch * left_size + lhs];
      value = Sum(value, Product(conjugate ? Conjugate(a) : a, right[batch * right_size + rhs]));
    }
    const T sum = Reduce(value, shared);
    if (threadIdx.x == 0) scratch[batch * capacity + chunk] = sum;
    __syncthreads();
    if (total - task <= gridDim.x) break;
    task += gridDim.x;
  }
}

template <typename T>
__global__ void Finish(const T* scratch, T* output, uint64_t chunks,
                       uint64_t capacity, uint64_t batches) {
  __shared__ T shared[kThreads];
  for (uint64_t batch = blockIdx.x; batch < batches;) {
    T value{};
    for (uint64_t i = threadIdx.x; i < chunks;) {
      value = Sum(value, scratch[batch * capacity + i]);
      if (chunks - i <= kThreads) break;
      i += kThreads;
    }
    const T sum = Reduce(value, shared);
    if (threadIdx.x == 0) output[batch] = Sum(output[batch], sum);
    __syncthreads();
    if (batches - batch <= gridDim.x) break;
    batch += gridDim.x;
  }
}

}  // namespace tensor0::stride::cuda::dot
