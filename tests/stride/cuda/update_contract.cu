// Standalone device-backed tests of the Update host boundary and alias contract.
#include "cuda/update.cu"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) {
  Require(status == cudaSuccess, cudaGetErrorString(status));
}

namespace native = tensor0::stride::cuda::update;
namespace layout = tensor0::stride::layout;
namespace descriptor = tensor0::stride::descriptor;
using State = native::UpdatePreparedState;
std::unique_ptr<State> Prepare(const std::vector<int64_t>& words) {
  const auto created = Tensor0StrideCudaUpdatePreparedCreatedCount();
  auto state = native::InstantiateUpdate({words.data(), words.size()});
  Require(state.has_value(), "valid Update plan rejected");
  Require(Tensor0StrideCudaUpdatePreparedCreatedCount() == created + 1,
          "instantiate did not create exactly one Update state");
  return std::move(*state);
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
    const auto packed = layout::PackOwnerFiber(
        descriptor::DecodeLayout(words.data(), words.size()), 1);
    descriptor_count = static_cast<int64_t>(packed.words.size());
    Cuda(cudaMemcpyAsync(descriptor.data, packed.words.data(), packed.words.size() * sizeof(int64_t),
                         cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  std::array<unsigned char, bytes> Snapshot() {
    std::array<unsigned char, bytes> values;
    Cuda(cudaMemcpyAsync(values.data(), arena, bytes, cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    return values;
  }
  ffi::Error Invoke(State* state) {
    return native::Update<ffi::F32, float, float>(
        state, ffi::AnyBuffer(&source),
        ffi::BufferR2<ffi::F32>(&base), ffi::AnyBuffer(&alpha), ffi::AnyBuffer(&beta),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::BufferR2<ffi::F32>(&result), stream);
  }
  void Reject(State* state, const std::string& message) {
    const auto before = Snapshot();
    const auto error = Invoke(state);
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos,
            "unexpected rejection: " + error.message());
    // Synchronize even after rejection to detect any prematurely submitted write.
    Cuda(cudaStreamSynchronize(stream));
    Require(Snapshot() == before, "rejected call modified device storage");
  }
  void Accept(State* state, const std::vector<int>& selected) {
    std::array<float, 8> src, old, actual;
    Cuda(cudaMemcpyAsync(src.data(), source.data, sizeof(src), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaMemcpyAsync(old.data(), base.data, sizeof(old), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    const auto error = Invoke(state);
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
  auto state = Prepare(h.words);
  auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr;
  Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), 8 * sizeof(float)));
  // No synchronization or host reads between the producer, Update and consumer.
  Produce<<<1, 8, 0, h.stream>>>(static_cast<float*>(h.source.data),
      static_cast<float*>(h.base.data), static_cast<float*>(h.alpha.data),
      static_cast<float*>(h.beta.data));
  Cuda(cudaGetLastError());
  const auto error = h.Invoke(state.get());
  Require(!error.failure(), error.message());
  const auto destroyed = Tensor0StrideCudaUpdatePreparedDestroyedCount();
  state.reset();
  Require(Tensor0StrideCudaUpdatePreparedDestroyedCount() == destroyed + 1,
          "Update state was not released before stream synchronization");
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
    auto state = Prepare(h.words);
    h.Accept(state.get(), {1, 3, 5});
  }
  {
    Harness h;
    h.shape[0] = 0;
    h.coefficient_count = 0;
    h.UploadDescriptor();
    const auto before = h.Snapshot();
    auto state = Prepare(h.words);
    const auto error = h.Invoke(state.get());
    Require(!error.failure(), error.message());
    Require(h.Snapshot() == before, "zero-batch update modified storage");
  }
  for (int alias = 0; alias < 3; ++alias) {
    Harness h;
    if (alias >= 1) h.result.data = h.base.data;
    if (alias == 2) h.source.data = h.base.data;
    h.UploadDescriptor();
    auto state = Prepare(h.words);
    h.Accept(state.get(), {1, 3, 5});
  }
  // CPU normalization ignores singleton strides only for nonempty records.
  for (const auto& words : std::vector<std::vector<int64_t>>{
           {1, 8, 8, 1, 2, 1, 1, 1, 3, 71, 2, -91, 2},
           {1, 8, 8, 1, 1, 5, 5, 3, -2, -2}}) {
    Harness h;
    h.source.data = h.result.data = h.base.data;
    h.words = words;
    h.UploadDescriptor();
    auto state = Prepare(h.words);
    h.Accept(state.get(), {1, 3, 5});
  }
  for (bool triple : {false, true}) {
    Harness h;
    if (triple) h.source.data = h.result.data = h.base.data;
    h.words = {1, 8, 8, 1, 2, 0, 0, 0, 1, 2, 7, 2, 7};
    h.UploadDescriptor();
    auto state = Prepare(h.words);
    h.Accept(state.get(), {});
  }
}

void ReuseAndHostLifetime() {
  std::unique_ptr<State> state;
  {
    auto words = std::vector<int64_t>{1, 8, 8, 1, 1, 1, 1, 3, 2, 2};
    state = Prepare(words);
    std::fill(words.begin(), words.end(), -123);
    Harness h;
    h.UploadDescriptor();
    h.Accept(state.get(), {1, 3, 5});
  }
  // The host words and the first arena have both expired. No device address
  // can be retained in the reusable plan.
  const auto created = Tensor0StrideCudaUpdatePreparedCreatedCount();
  Harness h;
  h.UploadDescriptor();
  for (int64_t batches : {3, 0, 2, 1}) {
    h.shape[0] = batches;
    h.coefficient_count = batches == 0 ? 0 : batches;
    const auto before = h.Snapshot();
    const auto error = h.Invoke(state.get());
    Require(!error.failure(), error.message());
    const auto after = h.Snapshot();
    if (batches == 0) {
      Require(before == after, "zero-batch reused Update wrote storage");
    } else {
      std::array<float, 24> source{}, base{}, result{};
      std::array<float, 3> alpha{}, beta{};
      const auto count = static_cast<size_t>(batches) * 8 * sizeof(float);
      std::memcpy(source.data(), before.data(), count);
      std::memcpy(base.data(), before.data() + 128, count);
      std::memcpy(alpha.data(), before.data() + 384, batches * sizeof(float));
      std::memcpy(beta.data(), before.data() + 400, batches * sizeof(float));
      std::memcpy(result.data(), after.data() + 256, count);
      for (int64_t batch = 0; batch < batches; ++batch) {
        for (int index = 0; index < 8; ++index) {
          const bool selected = index == 1 || index == 3 || index == 5;
          const auto element = batch * 8 + index;
          Require(result[element] == (selected ? alpha[batch] * source[element] +
                                                beta[batch] * base[element] : base[element]),
                  "reused Update batch output mismatch");
        }
      }
    }
  }
  h.shape[0] = h.coefficient_count = 1;
  --h.descriptor_count;
  h.Reject(state.get(), "descriptor operand shape");
  ++h.descriptor_count;
  h.Accept(state.get(), {1, 3, 5});
  Require(Tensor0StrideCudaUpdatePreparedCreatedCount() == created,
          "execute unexpectedly created another Update state");
}

// Reuse a single plan across runtime coefficient types, forms, values, batches,
// and storage aliasing; read expected values from before each execution.
void ReuseCoefficientsAndAliases() {
  Harness h;
  h.UploadDescriptor();
  auto state = Prepare(h.words);
  const auto created = Tensor0StrideCudaUpdatePreparedCreatedCount();
  for (int64_t batches : {1, 2, 0, 3}) {
    h.shape[0] = batches;
    for (bool integer : {false, true}) {
      h.alpha.dtype = h.beta.dtype = integer ? XLA_FFI_DataType_S32 : XLA_FFI_DataType_F32;
      for (int form = 0; form < 3; ++form) {
        h.alpha.rank = h.beta.rank = form == 0 ? 0 : 1;
        h.coefficient_count = form == 2 ? batches : 1;
        for (int mode = 0; mode < 3; ++mode) {
          const int a = mode == 0 ? 0 : mode == 1 ? 1 : 2;
          const int b = mode == 0 ? 1 : mode == 1 ? 0 : 3;
          for (int alias = 0; alias < 3; ++alias) {
            h.source.data = h.arena;
            h.base.data = h.arena + 128;
            h.result.data = h.arena + 256;
            if (alias >= 1) h.result.data = h.base.data;
            if (alias == 2) h.source.data = h.base.data;
            std::array<float, 24> src{}, old{};
            for (int i = 0; i < 24; ++i) {
              src[i] = float(i + 1);
              old[i] = float(40 + i);
            }
            Cuda(cudaMemcpyAsync(h.source.data, src.data(), sizeof(src), cudaMemcpyHostToDevice, h.stream));
            Cuda(cudaMemcpyAsync(h.base.data, old.data(), sizeof(old), cudaMemcpyHostToDevice, h.stream));
            const int coefficient_size = form == 2 ? std::max<int64_t>(batches, 1) : 1;
            std::array<int32_t, 3> ai{}, bi{};
            std::array<float, 3> af{}, bf{};
            for (int i = 0; i < coefficient_size; ++i) {
              ai[i] = a + (form == 2 && mode == 2 ? i : 0);
              bi[i] = b + (form == 2 && mode == 2 ? i : 0);
              af[i] = float(ai[i]); bf[i] = float(bi[i]);
            }
            Cuda(cudaMemcpyAsync(h.alpha.data, integer ? static_cast<const void*>(ai.data()) : af.data(),
                                 coefficient_size * 4, cudaMemcpyHostToDevice, h.stream));
            Cuda(cudaMemcpyAsync(h.beta.data, integer ? static_cast<const void*>(bi.data()) : bf.data(),
                                 coefficient_size * 4, cudaMemcpyHostToDevice, h.stream));
            const auto before = h.Snapshot();
            const auto error = h.Invoke(state.get());
            Require(!error.failure(), error.message());
            const auto after = h.Snapshot();
            if (batches == 0) {
              Require(after == before, "zero-batch Update wrote storage");
              continue;
            }
            const auto* input = static_cast<unsigned char*>(h.source.data) - h.arena + before.data();
            const auto* base = static_cast<unsigned char*>(h.base.data) - h.arena + before.data();
            const auto* output = static_cast<unsigned char*>(h.result.data) - h.arena + after.data();
            for (int i = 0; i < batches * 8; ++i) {
              float x, y, actual;
              std::memcpy(&x, input + i * 4, 4);
              std::memcpy(&y, base + i * 4, 4);
              std::memcpy(&actual, output + i * 4, 4);
              const int batch = i / 8;
              const bool touched = i % 8 == 1 || i % 8 == 3 || i % 8 == 5;
              const float expected = touched ? ai[form == 2 ? batch : 0] * x +
                  bi[form == 2 ? batch : 0] * y : y;
              Require(actual == expected, "reused Update coefficient/alias mismatch");
            }
          }
        }
      }
    }
  }
  h.shape[0] = 1;
  h.alpha.rank = h.beta.rank = 1;
  h.alpha.dtype = h.beta.dtype = XLA_FFI_DataType_F32;
  h.coefficient_count = 1;
  h.source.data = h.arena;
  h.base.data = h.arena + 128;
  h.result.data = h.source.data;
  const float zero = 0;
  Cuda(cudaMemcpyAsync(h.alpha.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
  h.Reject(state.get(), "requires identical source/base/result");
  h.result.data = h.arena + 256;
  const float alpha = 2, beta = 3;
  Cuda(cudaMemcpyAsync(h.alpha.data, &alpha, sizeof(alpha), cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.beta.data, &beta, sizeof(beta), cudaMemcpyHostToDevice, h.stream));
  h.Accept(state.get(), {1, 3, 5});
  Require(Tensor0StrideCudaUpdatePreparedCreatedCount() == created,
          "coefficient/alias reuse unexpectedly instantiated Update");
}

void InvalidStaticPlans() {
  const auto created = Tensor0StrideCudaUpdatePreparedCreatedCount();
  const auto destroyed = Tensor0StrideCudaUpdatePreparedDestroyedCount();
  auto reject = [&](Harness& h, const std::string& message) {
    const auto before = h.Snapshot();
    auto state = native::InstantiateUpdate({h.words.data(), h.words.size()});
    Require(!state.has_value(), "invalid static Update descriptor created a state: " + message);
    Require(state.error().message().find(message) != std::string::npos,
            "unexpected static rejection: " + state.error().message());
    Require(Tensor0StrideCudaUpdatePreparedCreatedCount() == created &&
            Tensor0StrideCudaUpdatePreparedDestroyedCount() == destroyed,
            "invalid static descriptor constructed then discarded an Update state");
    Require(h.Snapshot() == before, "invalid static descriptor modified device storage");
  };
  for (bool out_of_bounds : {false, true}) {
    Harness h;
    h.words = {1, 8, 8, 2, 1, 0, 0, 2, 1, 1,
               1, 2, out_of_bounds ? 8 : 2, 2, 1, out_of_bounds ? 1 : 0};
    reject(h, out_of_bounds ? "destination address" : "injective");
  }
  Harness h;
  h.words = {1, 8, 8, 1, 1, 0, 0, 3, std::numeric_limits<int64_t>::max(), 1};
  reject(h, "address arithmetic overflow");
}

void Rejections() {
  for (const auto& words : std::vector<std::vector<int64_t>>{
           {1, 8, 8, 1, 2, 0, 0, 2, 2, 1, 2, 2, 1},
           {1, 8, 8, 1, 2, 0, 0, 0, 1, 2, 7, 2, 9},
           {1, 8, 8, 1, 1, 0, 1, 0, 1, 1},
           {1, 8, 8, 2, 1, 0, 0, 2, 1, 1, 1, 2, 3, 2, 1, 1}}) {
    Harness h;
    h.words = words;
    h.source.data = h.result.data = h.base.data;
    h.UploadDescriptor();
    auto state = Prepare(h.words);
    h.Reject(state.get(), "identical per-element addresses");
    h.result.data = h.arena + 256;
    const auto valid = h.Invoke(state.get());
    Require(!valid.failure(), valid.message());
    Cuda(cudaStreamSynchronize(h.stream));
  }
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
    auto state = Prepare(h.words);
    h.Reject(state.get(), kind == 2 ? "requires identical source/base/result" : "overlap");
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
    auto state = Prepare(h.words);
    h.Reject(state.get(), message);
  }
  {
    Harness h;
    h.coefficient_count = 0;
    h.UploadDescriptor();
    auto state = Prepare(h.words);
    h.Reject(state.get(), "one value per batch");
  }
  {
    Harness h;
    h.result.data = h.source.data;
    h.UploadDescriptor();
    const float zero = 0;
    Cuda(cudaMemcpyAsync(h.alpha.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    auto state = Prepare(h.words);
    h.Reject(state.get(), "requires identical source/base/result");
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
      auto state = Prepare(h.words);
      h.Reject(state.get(), multiply ? "batch storage size overflows" : "buffer size exceeds addressable storage");
    }
  }

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
    const auto created = Tensor0StrideCudaUpdatePreparedCreatedCount();
    const auto destroyed = Tensor0StrideCudaUpdatePreparedDestroyedCount();
    InvalidStaticPlans();
    Successes();
    ReuseAndHostLifetime();
    ReuseCoefficientsAndAliases();
    Rejections();
    StreamDependency();
    Require(Tensor0StrideCudaUpdatePreparedCreatedCount() - created ==
            Tensor0StrideCudaUpdatePreparedDestroyedCount() - destroyed,
            "Update prepared state leaked");
    std::cout << "CUDA Update contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
