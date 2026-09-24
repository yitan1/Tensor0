#include "arithmetic.cuh"
#include "../ffi/errors.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ffi = xla::ffi;
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

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA dot: ") + cudaGetErrorString(status));
}

template <ffi::DataType Dtype, typename T>
ffi::Error Dot(ffi::Span<const int64_t> words, int64_t conjugate,
               ffi::AnyBuffer left, ffi::AnyBuffer right, ffi::BufferR1<ffi::S64> descriptor,
               ffi::ResultBufferR1<Dtype> result, ffi::ResultBufferR2<Dtype> scratch,
               cudaStream_t stream) {
  static_assert(sizeof(T) == ffi::ByteWidth(Dtype));
  return ContainErrors([&] {
    if (left.element_type() != Dtype || right.element_type() != Dtype) throw std::invalid_argument("CUDA dot supports only same-dtype F32/F64/C64/C128 inputs and result");
    if (conjugate != 0 && conjugate != 1) throw std::invalid_argument("CUDA dot conjugate_left must be zero or one");
    if (left.dimensions().size() != 2 || right.dimensions().size() != 2) throw std::invalid_argument("CUDA dot inputs must be rank-two");
    for (const auto dimensions : {left.dimensions(), right.dimensions(), result->dimensions(), scratch->dimensions()}) {
      for (auto dimension : dimensions) if (dimension < 0) throw std::invalid_argument("CUDA dot dimensions must be nonnegative");
    }
    if (descriptor.dimensions()[0] < 0 || static_cast<uint64_t>(descriptor.dimensions()[0]) != words.size()) throw std::invalid_argument("CUDA dot descriptor operand shape does not match layout");
    const auto decoded = descriptor::DecodeAddressLayout(words.begin(), words.size());
    const uint64_t batches = result->dimensions()[0];
    if (static_cast<uint64_t>(left.dimensions()[0]) != batches || static_cast<uint64_t>(right.dimensions()[0]) != batches ||
        static_cast<uint64_t>(left.dimensions()[1]) != decoded.source_size || static_cast<uint64_t>(right.dimensions()[1]) != decoded.output_size ||
        static_cast<uint64_t>(scratch->dimensions()[0]) != batches) throw std::invalid_argument("CUDA dot buffer dimensions do not match layout");
    std::vector<uint64_t> counts;
    uint64_t capacity = 0;
    for (const auto& record : decoded.records) {
      const auto count = layout::ElementCount(record);
      uint64_t total;
      if (!layout::CheckedMultiply(count, batches, &total)) throw std::invalid_argument("CUDA dot logical batch size overflows");
      counts.push_back(count);
      capacity = std::max(capacity, count == 0 ? 0 : (count - 1) / kChunk + 1);
    }
    if (static_cast<uint64_t>(scratch->dimensions()[1]) != capacity) throw std::invalid_argument("CUDA dot scratch capacity does not match layout");
    uint64_t left_elements, right_elements, scratch_elements;
    if (!layout::CheckedMultiply(batches, decoded.source_size, &left_elements) ||
        !layout::CheckedMultiply(batches, decoded.output_size, &right_elements) ||
        !layout::CheckedMultiply(batches, capacity, &scratch_elements)) throw std::invalid_argument("CUDA dot batch storage size overflows");
    const auto output_bytes = BufferBytes(batches, sizeof(T));
    const auto scratch_bytes = BufferBytes(scratch_elements, sizeof(T));
    const auto left_bytes = BufferBytes(left_elements, sizeof(T));
    const auto right_bytes = BufferBytes(right_elements, sizeof(T));
    const auto descriptor_bytes = BufferBytes(words.size(), sizeof(int64_t));
    auto* output = reinterpret_cast<T*>(result->untyped_data());
    auto* temporary = reinterpret_cast<T*>(scratch->untyped_data());
    for (const auto target : {std::pair<void*, uint64_t>{output, output_bytes}, {temporary, scratch_bytes}}) {
      ValidateDisjointBuffers(left.untyped_data(), left_bytes, target.first, target.second);
      ValidateDisjointBuffers(right.untyped_data(), right_bytes, target.first, target.second);
      ValidateDisjointBuffers(descriptor.typed_data(), descriptor_bytes, target.first, target.second);
    }
    ValidateDisjointBuffers(output, output_bytes, temporary, scratch_bytes);
    // All metadata, sizes and aliases are validated before either result is written.
    if (output_bytes != 0) Check(cudaMemsetAsync(output, 0, output_bytes, stream));
    std::size_t cursor = 4;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const auto count = counts[i];
      if (count != 0 && batches != 0) {
        const uint64_t chunks = (count - 1) / kChunk + 1;
        const uint64_t tasks = batches * chunks;
        Partials<T><<<static_cast<unsigned>(std::min<uint64_t>(tasks, 65535)), kThreads, 0, stream>>>(
            reinterpret_cast<const T*>(left.untyped_data()), reinterpret_cast<const T*>(right.untyped_data()),
            temporary, descriptor.typed_data() + cursor, count, chunks, capacity, tasks,
            decoded.source_size, decoded.output_size, conjugate != 0);
        Check(cudaGetLastError());
        Finish<T><<<static_cast<unsigned>(std::min<uint64_t>(batches, 65535)), kThreads, 0, stream>>>(temporary, output, chunks, capacity, batches);
        Check(cudaGetLastError());
      }
      cursor += 3 + 3 * decoded.records[i].shape.size();
    }
  });
}
}  // namespace tensor0::stride::cuda::dot

#define TENSOR0_CUDA_DOT(Suffix, Dtype, T) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCudaDot##Suffix##V1, (tensor0::stride::cuda::dot::Dot<ffi::Dtype, T>), \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout").Attr<int64_t>("conjugate_left") \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::BufferR1<ffi::S64>>() \
        .Ret<ffi::BufferR1<ffi::Dtype>>().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::PlatformStream<cudaStream_t>>()); \
extern "C" void* Tensor0StrideCudaDot##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCudaDot##Suffix##V1); \
}
TENSOR0_CUDA_DOT(F32, F32, float)
TENSOR0_CUDA_DOT(F64, F64, double)
TENSOR0_CUDA_DOT(C64, C64, tensor0::stride::cuda::arithmetic::Complex<float>)
TENSOR0_CUDA_DOT(C128, C128, tensor0::stride::cuda::arithmetic::Complex<double>)
#undef TENSOR0_CUDA_DOT
