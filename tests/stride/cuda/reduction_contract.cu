// Device-backed reduction validation and non-default-stream ordering.
#include "cuda/reduction.cu"

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
  std::vector<int64_t> words{1, 8, 8, 1, 2, 0, 0, 2, 2, 2, 1, 1, 2, 99, 1, 1, 0};
  std::vector<int64_t> indices{0};
  std::array<int64_t, 2> shape{1, 8};
  int64_t coefficient_count = 1, descriptor_count = 0;
  XLA_FFI_Buffer source{}, result{}, coefficient{}, descriptor{};
  XLA_FFI_Buffer Buffer(size_t offset, XLA_FFI_DataType dtype, int64_t rank, int64_t* dims) {
    return {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, arena + offset, rank, dims};
  }
  Harness() {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    source = Buffer(0, XLA_FFI_DataType_F32, 2, shape.data());
    result = Buffer(256, XLA_FFI_DataType_F32, 2, shape.data());
    coefficient = Buffer(384, XLA_FFI_DataType_F32, 1, &coefficient_count);
    descriptor = Buffer(512, XLA_FFI_DataType_S64, 1, &descriptor_count);
    std::array<float, bytes / sizeof(float)> initial{};
    initial.fill(99);
    for (int i = 0; i < 8; ++i) initial[i] = float(i + 1);
    initial[96] = 2;
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
    XLA_FFI_ArgType type = XLA_FFI_ArgType_BUFFER;
    void* pointer = &coefficient;
    XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, 1, &type, &pointer};
    return tensor0::stride::cuda::reduction::Reduction<ffi::F32, float, float>(
        {words.data(), words.size()}, {indices.data(), indices.size()}, ffi::AnyBuffer(&source),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::RemainingArgs(&args, 0),
        ffi::BufferR2<ffi::F32>(&result), stream);
  }
  void Reject(const std::string& message) {
    const auto before = Snapshot();
    const auto error = Invoke();
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos, "unexpected rejection: " + error.message());
    Require(Snapshot() == before, "rejected invocation changed device storage");
  }
  void Expect(std::array<float, 8> expected) {
    const auto error = Invoke();
    Require(!error.failure(), error.message());
    std::array<float, 8> actual;
    Cuda(cudaMemcpyAsync(actual.data(), result.data, sizeof(actual), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    Require(actual == expected, "wrong accumulated output");
  }
};

void Rejections() {
  for (int kind = 0; kind < 5; ++kind) {
    Harness h;
    if (kind == 0) h.result.data = h.source.data;
    if (kind == 1) h.result.data = static_cast<float*>(h.source.data) + 1;
    if (kind == 2) h.coefficient.data = h.result.data;
    if (kind == 3) h.descriptor.data = h.result.data;
    if (kind == 4) h.result.data = static_cast<int64_t*>(h.descriptor.data) + 1;
    h.Upload(); h.Reject("overlap");
  }
  for (int kind = 0; kind < 10; ++kind) {
    Harness h; h.Upload();
    std::string message;
    switch (kind) {
      case 0: h.source.rank = 1; message = "rank-two"; break;
      case 1: --h.descriptor_count; message = "descriptor operand shape"; break;
      case 2: h.coefficient_count = 2; message = "batch count"; break;
      case 3: h.source.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 4: h.coefficient.dtype = XLA_FFI_DataType_S64; message = "coefficient dtype"; break;
      case 5: h.indices = {-1}; message = "record indices"; break;
      case 6: h.indices.clear(); message = "record count"; break;
      case 7: h.shape[0] = -1; message = "nonnegative"; break;
      case 8: h.words[1] = 9; message = "dimensions do not match"; break;
      case 9: h.words[1] = -1; message = "dimensions do not match"; break;
    }
    h.Reject(message);
  }
  for (int kind = 0; kind < 12; ++kind) {
    Harness h;
    std::string message;
    switch (kind) {
      case 0: h.words[15] = 2; message = "not boolean"; break;
      case 1: h.words[11] = 2; message = "axis roles"; break;
      case 2: h.words[14] = 0; message = "zero nontrivial stride"; break;
      case 3: h.words[6] = 8; message = "address exceeds"; break;
      case 4: h.words[5] = -1; message = "offset exceeds"; break;
      case 5: h.words[9] = INT64_MAX; message = "address"; break;
      case 6: h.words.pop_back(); message = "rank exceeds"; break;
      case 7: h.words.push_back(0); message = "trailing"; break;
      case 8: h.words = {1,8,8,1,2,0,0,2,2,2,1,2,2,1,1,0,0}; message = "injective"; break;
      case 9: h.words = {1,8,8,1,1,0,8,0,0,1,0,1}; message = "address exceeds"; break;
      case 10: h.words = {1,8,8,1,2,0,0,4294967296LL,4294967296LL,0,0,1,1,0,0,1,1}; message = "count overflow"; break;
      case 11: h.words = {1,8,8,1,1,0,0,-1,0,1,0,1}; h.shape[0] = 2; message = "logical batch size overflows"; break;
    }
    h.Upload(); h.Reject(message);
  }
  {
    // An invalid later record must fail before the earlier valid one writes.
    Harness h;
    h.words[3] = 2;
    h.words.insert(h.words.end(), {1, 0, 8, 1, 1, 1, 1, 0});
    h.Upload(); h.Reject("address exceeds");
  }
  for (bool trailing : {false, true}) {
    // Original-axis prefix overflow must precede even a trailing-word error.
    Harness h;
    h.words = {1,8,8,1,3,0,0,-1,2,0,0,0,0,1,2,1,0,1,0,1,0,1};
    if (trailing) h.words.push_back(0);
    h.Upload(); h.Reject("layout element count overflows");
  }
  for (bool multiply : {false, true}) {
    Harness h;
    h.shape = {multiply ? 3 : 1, multiply ? std::numeric_limits<int64_t>::max() : std::numeric_limits<ptrdiff_t>::max() / 4 + 1};
    h.words = {1,h.shape[1],h.shape[1],0};
    h.Upload(); h.Reject(multiply ? "batch storage size overflows" : "buffer size exceeds addressable storage");
  }
}

__global__ void Produce(float* source, float* coefficient) {
  const auto start = clock64();
  while (clock64() - start < 1000000) {}
  source[threadIdx.x] = float(threadIdx.x + 1);
  if (threadIdx.x == 0) *coefficient = 3;
}
__global__ void Consume(const float* result, float* observed) { observed[threadIdx.x] = result[threadIdx.x] + 1; }
void StreamDependency() {
  Harness h; h.Upload();
  auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr;
  Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), 8 * sizeof(float)));
  Produce<<<1,8,0,h.stream>>>(static_cast<float*>(h.source.data), static_cast<float*>(h.coefficient.data));
  const auto error = h.Invoke(); Require(!error.failure(), error.message());
  Consume<<<1,8,0,h.stream>>>(static_cast<float*>(h.result.data), observed);
  Cuda(cudaMemcpyAsync(host, observed, 8 * sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  const std::array<float,8> expected{13,19,1,1,1,1,1,1};
  Require(std::equal(expected.begin(), expected.end(), host), "stream dependency failure");
  Cuda(cudaFreeHost(host));
}

void Successes() {
  { Harness h; h.Upload(); h.Expect({8,12,0,0,0,0,0,0}); }
  { Harness h; h.words = {1,8,8,1,1,0,0,0,1,1,99,1}; h.Upload(); h.Expect({}); }
  {
    Harness h; h.shape[0] = 0; h.coefficient_count = 0; h.Upload();
    const auto before = h.Snapshot(); const auto error = h.Invoke();
    Require(!error.failure(), error.message()); Require(h.Snapshot() == before, "zero batch write");
  }
  {
    // UINT64_MAX contributions with zero coefficient must not read or iterate.
    Harness h; h.words = {1,8,8,1,1,0,0,-1,0,1,99,1};
    h.Upload(); float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Expect({});
  }
}
}  // namespace

int main() {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    std::cout << "CUDA device unavailable\n"; return 77;
  }
  try {
    Cuda(status); Rejections(); Successes(); StreamDependency();
    std::cout << "CUDA Reduction contract passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
