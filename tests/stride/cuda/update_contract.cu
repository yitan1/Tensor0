// Standalone device-backed tests of the Update host boundary and alias contract.
#include "cuda/update.cu"

#include <array>
#include <iostream>
#include <limits>

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) {
  Require(status == cudaSuccess, cudaGetErrorString(status));
}

struct Harness {
  // One allocation makes partial overlap tests real, including descriptor overlap.
  static constexpr size_t bytes = 4096;
  unsigned char* arena = nullptr;
  cudaStream_t stream = nullptr;
  std::vector<int64_t> words{1, 8, 8, 1, 1, 1, 1, 3, 2, 2};
  std::array<int64_t, 2> shape{1, 8};
  int64_t coefficient_count = 1, descriptor_count = 0;
  XLA_FFI_Buffer source{}, base{}, result{}, alpha{}, beta{}, descriptor{};

  XLA_FFI_Buffer Buffer(size_t offset, XLA_FFI_DataType dtype,
                        int64_t rank, int64_t* dimensions) {
    return {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, arena + offset, rank, dimensions};
  }
  Harness() {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    source = Buffer(0, XLA_FFI_DataType_F32, 2, shape.data());
    base = Buffer(128, XLA_FFI_DataType_F32, 2, shape.data());
    result = Buffer(256, XLA_FFI_DataType_F32, 2, shape.data());
    alpha = Buffer(384, XLA_FFI_DataType_F32, 1, &coefficient_count);
    beta = Buffer(400, XLA_FFI_DataType_F32, 1, &coefficient_count);
    descriptor = Buffer(512, XLA_FFI_DataType_S64, 1, &descriptor_count);
    std::array<float, bytes / sizeof(float)> initial{};
    initial.fill(99);
    for (int i = 0; i < 8; ++i) {
      initial[i] = float(i + 1);
      initial[32 + i] = float(10 + i);
    }
    initial[96] = 2;
    initial[100] = 3;
    Cuda(cudaMemcpyAsync(arena, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  ~Harness() {
    cudaStreamDestroy(stream);
    cudaFree(arena);
  }
  void UploadDescriptor() {
    descriptor_count = static_cast<int64_t>(words.size());
    Cuda(cudaMemcpyAsync(descriptor.data, words.data(), words.size() * sizeof(int64_t),
                         cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  std::array<unsigned char, bytes> Snapshot() {
    std::array<unsigned char, bytes> values;
    Cuda(cudaMemcpyAsync(values.data(), arena, bytes, cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    return values;
  }
  ffi::Error Invoke() {
    return tensor0::stride::cuda::update::Update<ffi::F32, float, float>(
        ffi::Span<const int64_t>(words.data(), words.size()), ffi::AnyBuffer(&source),
        ffi::BufferR2<ffi::F32>(&base), ffi::AnyBuffer(&alpha), ffi::AnyBuffer(&beta),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::BufferR2<ffi::F32>(&result), stream);
  }
  void Reject(const std::string& message) {
    const auto before = Snapshot();
    const auto error = Invoke();
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos,
            "unexpected rejection: " + error.message());
    // Synchronize even after rejection to detect any prematurely submitted write.
    Cuda(cudaStreamSynchronize(stream));
    Require(Snapshot() == before, "rejected call modified device storage");
  }
  void Accept(const std::vector<int>& selected) {
    std::array<float, 8> src, old, actual;
    Cuda(cudaMemcpyAsync(src.data(), source.data, sizeof(src), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaMemcpyAsync(old.data(), base.data, sizeof(old), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    const auto error = Invoke();
    Require(!error.failure(), error.message());
    Cuda(cudaStreamSynchronize(stream));
    Cuda(cudaMemcpy(actual.data(), result.data, sizeof(actual), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 8; ++i) {
      const bool touched = std::find(selected.begin(), selected.end(), i) != selected.end();
      Require(actual[i] == (touched ? 2 * src[i] + 3 * old[i] : old[i]),
              "wrong update or base hole at " + std::to_string(i));
    }
  }
};

__global__ void Produce(float* source, float* base, float* alpha, float* beta) {
  // Keep the producer pending long enough to expose accidental default-stream use.
  const auto start = clock64();
  while (clock64() - start < 1000000) {}
  const int i = threadIdx.x;
  source[i] = float(20 + i);
  base[i] = float(40 + i);
  if (i == 0) { *alpha = 2; *beta = 3; }
}

__global__ void Consume(const float* result, float* observed) {
  observed[threadIdx.x] = result[threadIdx.x] + 1;
}

void StreamDependency() {
  Harness h;
  h.UploadDescriptor();
  auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr;
  Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), 8 * sizeof(float)));
  // No synchronization or host reads between the producer, Update and consumer.
  Produce<<<1, 8, 0, h.stream>>>(static_cast<float*>(h.source.data),
      static_cast<float*>(h.base.data), static_cast<float*>(h.alpha.data),
      static_cast<float*>(h.beta.data));
  Cuda(cudaGetLastError());
  const auto error = h.Invoke();
  Require(!error.failure(), error.message());
  Consume<<<1, 8, 0, h.stream>>>(static_cast<const float*>(h.result.data), observed);
  Cuda(cudaGetLastError());
  Cuda(cudaMemcpyAsync(host, observed, 8 * sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  for (int i = 0; i < 8; ++i) {
    const bool touched = i == 1 || i == 3 || i == 5;
    Require(host[i] == (touched ? 2 * (20 + i) + 3 * (40 + i) : 40 + i) + 1,
            "producer/Update/consumer stream dependency failed");
  }
  Cuda(cudaFreeHost(host));
}

void Successes() {
  {
    Harness h;
    // Read-only inputs may overlap when the result is disjoint from both.
    h.base.data = static_cast<float*>(h.source.data) + 1;
    h.UploadDescriptor();
    h.Accept({1, 3, 5});
  }
  {
    Harness h;
    h.shape[0] = 0;
    h.coefficient_count = 0;
    h.UploadDescriptor();
    const auto before = h.Snapshot();
    const auto error = h.Invoke();
    Require(!error.failure(), error.message());
    Require(h.Snapshot() == before, "zero-batch update modified storage");
  }
  for (int alias = 0; alias < 3; ++alias) {
    Harness h;
    if (alias >= 1) h.result.data = h.base.data;
    if (alias == 2) h.source.data = h.base.data;
    h.UploadDescriptor();
    h.Accept({1, 3, 5});
  }
  // CPU normalization ignores singleton strides only for nonempty records.
  for (const auto& words : std::vector<std::vector<int64_t>>{
           {1, 8, 8, 1, 2, 1, 1, 1, 3, 71, 2, -91, 2},
           {1, 8, 8, 1, 1, 5, 5, 3, -2, -2}}) {
    Harness h;
    h.source.data = h.result.data = h.base.data;
    h.words = words;
    h.UploadDescriptor();
    h.Accept({1, 3, 5});
  }
  for (bool triple : {false, true}) {
    Harness h;
    if (triple) h.source.data = h.result.data = h.base.data;
    h.words = {1, 8, 8, 1, 2, 0, 0, 0, 1, 2, 7, 2, 7};
    h.UploadDescriptor();
    h.Accept({});
  }
}

void Rejections() {
  for (int kind = 0; kind < 7; ++kind) {
    Harness h;
    if (kind == 0) h.result.data = static_cast<float*>(h.source.data) + 1;
    if (kind == 1) h.result.data = static_cast<float*>(h.base.data) + 1;
    if (kind == 2) h.result.data = h.source.data;
    if (kind == 3) h.alpha.data = h.result.data;
    if (kind == 4) h.beta.data = static_cast<float*>(h.result.data) + 1;
    if (kind == 5) h.descriptor.data = h.result.data;
    if (kind == 6) h.result.data = static_cast<int64_t*>(h.descriptor.data) + 1;
    h.UploadDescriptor();
    h.Reject(kind == 2 ? "requires identical source/base/result" : "overlap");
  }
  for (const auto& words : std::vector<std::vector<int64_t>>{
           // Transpose permutation, and strict empty identity (singleton included).
           {1, 8, 8, 1, 2, 0, 0, 2, 2, 1, 2, 2, 1},
           {1, 8, 8, 1, 2, 0, 0, 0, 1, 2, 7, 2, 9},
           {1, 8, 8, 1, 1, 0, 1, 0, 1, 1},
           // Valid first record must not run before an invalid later record.
           {1, 8, 8, 2, 1, 0, 0, 2, 1, 1, 1, 2, 3, 2, 1, 1}}) {
    Harness h;
    h.source.data = h.result.data = h.base.data;
    h.words = words;
    h.UploadDescriptor();
    h.Reject("identical per-element addresses");
  }
  for (int kind = 0; kind < 7; ++kind) {
    Harness h;
    h.UploadDescriptor();
    std::array<int64_t, 2> bad_shape{1, 7};
    std::string message;
    switch (kind) {
      case 0: h.source.rank = 1; message = "rank-two"; break;
      case 1: h.base.dims = bad_shape.data(); message = "dimensions"; break;
      case 2: --h.descriptor_count; message = "descriptor operand shape"; break;
      case 3: h.coefficient_count = 2; message = "one value per batch"; break;
      case 4: h.source.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 5: h.alpha.dtype = XLA_FFI_DataType_S64; message = "coefficient dtype"; break;
      case 6: h.beta.dtype = XLA_FFI_DataType_S64; message = "coefficient dtype"; break;
    }
    h.Reject(message);
  }
  {
    Harness h;
    h.coefficient_count = 0;
    h.UploadDescriptor();
    h.Reject("one value per batch");
  }
  {
    Harness h;
    h.result.data = h.source.data;
    h.UploadDescriptor();
    const float zero = 0;
    Cuda(cudaMemcpyAsync(h.alpha.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Reject("requires identical source/base/result");
  }
  for (bool out_of_bounds : {false, true}) {
    Harness h;
    h.words = {1, 8, 8, 2, 1, 0, 0, 2, 1, 1,
               1, 2, out_of_bounds ? 8 : 2, 2, 1, out_of_bounds ? 1 : 0};
    h.UploadDescriptor();
    h.Reject(out_of_bounds ? "destination address" : "injective");
  }
  // Deliberately oversized metadata must fail without accessing the tiny arena.
  // Exercise both operands of the checked storage multiplication/byte conversion.
  for (bool output : {false, true}) {
    for (bool multiply : {false, true}) {
      Harness h;
      const int64_t batches = multiply ? 3 : 1;
      const int64_t huge = multiply ? std::numeric_limits<int64_t>::max()
          : std::numeric_limits<std::ptrdiff_t>::max() / sizeof(float) + 1;
      std::array<int64_t, 2> source_shape{batches, output ? 8 : huge};
      std::array<int64_t, 2> output_shape{batches, output ? huge : 8};
      h.source.dims = source_shape.data();
      h.base.dims = h.result.dims = output_shape.data();
      h.words = {1, source_shape[1], output_shape[1], 0};
      h.UploadDescriptor();
      h.Reject(multiply ? "batch storage size overflows" : "buffer size exceeds addressable storage");
    }
  }
  Harness h;
  h.words = {1, 8, 8, 1, 1, 0, 0, 3, std::numeric_limits<int64_t>::max(), 1};
  h.UploadDescriptor();
  h.Reject("address arithmetic overflow");
}
}  // namespace

int main() {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) {
    std::cout << "CUDA device unavailable\n";
    return 77;
  }
  try {
    Cuda(status);
    Successes();
    Rejections();
    StreamDependency();
    std::cout << "CUDA Update contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
