// Device-backed Dot scratch/alias validation and non-default-stream ordering.
#include "cuda/dot.cu"
#include <array>
#include <iostream>
#include <limits>

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
struct Harness {
  static constexpr size_t bytes = 4096;
  unsigned char* arena = nullptr;
  cudaStream_t stream = nullptr;
  std::vector<int64_t> words{1, 8, 8, 1, 1, 0, 0, 8, 1, 1};
  std::array<int64_t, 2> shape{1, 8}, scratch_shape{1, 1};
  int64_t batches = 1, descriptor_count = 0, conjugate = 0;
  XLA_FFI_Buffer left{}, right{}, result{}, scratch{}, descriptor{};
  XLA_FFI_Buffer Buffer(size_t offset, XLA_FFI_DataType dtype, int64_t rank, int64_t* dims) {
    return {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, arena + offset, rank, dims};
  }
  Harness() {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    left = Buffer(0, XLA_FFI_DataType_F32, 2, shape.data());
    right = Buffer(128, XLA_FFI_DataType_F32, 2, shape.data());
    result = Buffer(256, XLA_FFI_DataType_F32, 1, &batches);
    scratch = Buffer(384, XLA_FFI_DataType_F32, 2, scratch_shape.data());
    descriptor = Buffer(512, XLA_FFI_DataType_S64, 1, &descriptor_count);
    std::array<float, bytes / sizeof(float)> initial{};
    initial.fill(99);
    for (int i = 0; i < 8; ++i) { initial[i] = float(i + 1); initial[32 + i] = 2; }
    Cuda(cudaMemcpyAsync(arena, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  ~Harness() { cudaStreamDestroy(stream); cudaFree(arena); }
  void Upload() {
    descriptor_count = words.size();
    Cuda(cudaMemcpyAsync(descriptor.data, words.data(), words.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  std::array<unsigned char, bytes> Snapshot() {
    std::array<unsigned char, bytes> values;
    Cuda(cudaMemcpyAsync(values.data(), arena, bytes, cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    return values;
  }
  ffi::Error Invoke() {
    return tensor0::stride::cuda::dot::Dot<ffi::F32, float>(
        {words.data(), words.size()}, conjugate, ffi::AnyBuffer(&left), ffi::AnyBuffer(&right),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::BufferR1<ffi::F32>(&result),
        ffi::BufferR2<ffi::F32>(&scratch), stream);
  }
  void Reject(const std::string& message) {
    const auto before = Snapshot(); const auto error = Invoke();
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos, "unexpected rejection: " + error.message());
    Require(Snapshot() == before, "rejected invocation changed device storage");
  }
  void Expect(float expected) {
    const auto error = Invoke(); Require(!error.failure(), error.message());
    float actual;
    Cuda(cudaMemcpyAsync(&actual, result.data, sizeof(actual), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream)); Require(actual == expected, "wrong Dot output");
  }
};
void Rejections() {
  for (int kind = 0; kind < 8; ++kind) {
    Harness h;
    if (kind == 0) h.result.data = h.left.data;
    if (kind == 1) h.result.data = static_cast<float*>(h.right.data) + 1;
    if (kind == 2) h.result.data = h.scratch.data;
    if (kind == 3) h.descriptor.data = h.result.data;
    if (kind == 4) h.scratch.data = h.left.data;
    if (kind == 5) h.scratch.data = static_cast<float*>(h.right.data) + 1;
    if (kind == 6) h.descriptor.data = h.scratch.data;
    if (kind == 7) h.scratch.data = static_cast<int64_t*>(h.descriptor.data) + 1;
    h.Upload(); h.Reject("overlap");
  }
  { Harness h; h.Upload(); h.conjugate = 2; h.Reject("zero or one"); }
  { Harness h; h.Upload(); h.left.dtype = XLA_FFI_DataType_F64; h.Reject("same-dtype"); }
  { Harness h; h.Upload(); h.descriptor_count--; h.Reject("descriptor operand shape"); }
  { Harness h; h.Upload(); h.scratch_shape[1] = 2; h.Reject("scratch capacity"); }
  { Harness h; h.Upload(); h.scratch_shape[0] = 2; h.Reject("dimensions"); }
  { Harness h; h.words[7] = 9; h.Upload(); h.Reject("address"); }
  { Harness h; h.words = {1,8,8,1,2,0,0,4294967296LL,4294967296LL,0,0,0,0}; h.Upload(); h.Reject("count"); }
  { Harness h; h.words = {1,8,8,1,2,0,0,4294967295LL,4294967297LL,0,0,0,0}; h.shape[0] = h.batches = h.scratch_shape[0] = 2; h.Upload(); h.Reject("logical batch size overflows"); }
  { Harness h; h.words = {1,0,0,0}; h.shape[1] = 0; h.shape[0] = h.batches = h.scratch_shape[0] = INT64_MAX; h.scratch_shape[1] = 0; h.Upload(); h.Reject("addressable storage"); }
  { Harness h; h.words = {1,INT64_MAX,8,0}; h.shape[1] = INT64_MAX; h.Upload(); h.Reject("dimensions"); }
  { Harness h; h.words = {1,8,8,0}; h.shape[0] = h.batches = h.scratch_shape[0] = INT64_MAX; h.scratch_shape[1] = 0; h.Upload(); h.Reject("batch storage size overflows"); }
}
__global__ void Produce(float* left, float* right) { left[threadIdx.x] = threadIdx.x + 1; right[threadIdx.x] = 3; }
__global__ void Consume(const float* result, float* observed) { *observed = *result + 1; }
void StreamDependency() {
  Harness h; h.Upload(); auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr; Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), sizeof(float)));
  Produce<<<1,8,0,h.stream>>>(static_cast<float*>(h.left.data), static_cast<float*>(h.right.data));
  const auto error = h.Invoke(); Require(!error.failure(), error.message());
  Consume<<<1,1,0,h.stream>>>(static_cast<float*>(h.result.data), observed);
  Cuda(cudaMemcpyAsync(host, observed, sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream)); Require(*host == 109, "stream dependency failure");
  Cuda(cudaFreeHost(host));
}
void Successes() {
  { Harness h; h.Upload(); h.Expect(72); }
  { Harness h; h.right.data = h.left.data; h.Upload(); h.Expect(204); }
  { Harness h; h.words = {1,8,8,1,1,0,0,0,1,1}; h.scratch_shape[1] = 0; h.Upload(); h.Expect(0); }
  { Harness h; h.words = {1,8,8,0}; h.scratch_shape[1] = 0; h.Upload(); h.Expect(0); }
  { Harness h; h.shape[0] = h.batches = h.scratch_shape[0] = 0; h.Upload(); const auto before = h.Snapshot(); const auto error = h.Invoke(); Require(!error.failure(), error.message()); Require(h.Snapshot() == before, "zero batch write"); }
}
}
int main() {
  int devices = 0; const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) { std::cout << "CUDA device unavailable\n"; return 77; }
  try { Cuda(status); Rejections(); Successes(); StreamDependency(); std::cout << "CUDA Dot contract passed\n"; return 0; }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
