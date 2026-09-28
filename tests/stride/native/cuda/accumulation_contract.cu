// Device-backed accumulation validation and non-default-stream ordering.
#include "cuda/accumulation.cu"

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>

namespace {
namespace native = tensor0::stride::cuda::accumulation;
using State = native::AccumulationPreparedState;

void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
std::unique_ptr<State> Prepare(const std::vector<int64_t>& words,
                               const std::vector<int64_t>& records) {
  const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
  auto state = native::InstantiateAccumulation({words.data(), words.size()},
                                                {records.data(), records.size()});
  Require(state.has_value(), "valid Accumulation plan rejected: " +
          (state.has_value() ? std::string{} : state.error().message()));
  Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() == created + 1,
          "instantiate did not create exactly one Accumulation state");
  return std::move(*state);
}

struct Harness {
  static constexpr size_t bytes = 4096;
  unsigned char* arena = nullptr;
  cudaStream_t stream = nullptr;
  std::vector<int64_t> words{1, 8, 8, 1, 2, 0, 0, 2, 2, 2, 1, 2, 1};
  std::vector<int64_t> indices{0};
  int64_t operand_count = 1;
  std::array<int64_t, 2> shape{1, 8};
  std::array<int64_t, 2> scratch_shape{1, 0};
  int64_t coefficient_count = 1, descriptor_count = 0;
  XLA_FFI_Buffer source{}, result{}, scratch{}, coefficient{}, second{}, descriptor{};
  XLA_FFI_Buffer Buffer(size_t offset, XLA_FFI_DataType dtype, int64_t rank, int64_t* dims) {
    return {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, arena + offset, rank, dims};
  }
  Harness() {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    source = Buffer(0, XLA_FFI_DataType_F32, 2, shape.data());
    result = Buffer(256, XLA_FFI_DataType_F32, 2, shape.data());
    scratch = Buffer(768, XLA_FFI_DataType_F32, 2, scratch_shape.data());
    coefficient = Buffer(384, XLA_FFI_DataType_F32, 1, &coefficient_count);
    second = Buffer(400, XLA_FFI_DataType_F32, 1, &coefficient_count);
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
    std::vector<int64_t> metadata;
    try {
      metadata = tensor0::stride::layout::PackOwnerFiber(tensor0::stride::descriptor::DecodeAccumulationLayout(words.data(), words.size())).words;
    } catch (const std::invalid_argument&) {
      // Static-rejection fixtures never execute; upload arbitrary owned bytes.
      metadata = words;
    }
    descriptor_count = metadata.size();
    Cuda(cudaMemcpyAsync(descriptor.data, metadata.data(), metadata.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  std::array<unsigned char, bytes> Snapshot() {
    std::array<unsigned char, bytes> values;
    Cuda(cudaMemcpyAsync(values.data(), arena, bytes, cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    return values;
  }
  ffi::Error Invoke(const State* state) {
    scratch_shape[0] = shape[0];
    XLA_FFI_ArgType types[2]{XLA_FFI_ArgType_BUFFER, XLA_FFI_ArgType_BUFFER};
    void* pointers[2]{&coefficient, &second};
    XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, operand_count, types, pointers};
    return native::Accumulation<ffi::F32, float, float>(
        state, ffi::AnyBuffer(&source), ffi::BufferR1<ffi::S64>(&descriptor),
        ffi::RemainingArgs(&args, 0), ffi::BufferR2<ffi::F32>(&result),
        ffi::BufferR2<ffi::F32>(&scratch), stream);
  }
  void Reject(const State* state, const std::string& message) {
    const auto before = Snapshot();
    const auto error = Invoke(state);
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos, "unexpected rejection: " + error.message());
    Require(Snapshot() == before, "rejected invocation changed device storage");
  }
  void Expect(const State* state, std::array<float, 8> expected) {
    const auto error = Invoke(state);
    Require(!error.failure(), error.message());
    std::array<float, 8> actual;
    Cuda(cudaMemcpyAsync(actual.data(), result.data, sizeof(actual), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    Require(actual == expected, "wrong accumulated output");
  }
};

void Classifier() {
  using namespace tensor0::stride;
  const std::vector<int64_t> words{1,6,8,2,
      2,0,2,2,3,3,1,0,-1,
      2,0,0,2,3,3,1,3,1};
  auto prepared = Prepare(words, {});
  Require(prepared->counts == std::vector<uint64_t>({6,6}), "cached record counts");
  const auto& cached = prepared->schedules;
  Require(cached.size() == 2 && cached[0].owners == 3 && cached[0].contributions == 2 &&
          cached[1].owners == 6 && cached[1].contributions == 1,
          "cached owned-fiber and injective schedules");
  for (const auto& destination : std::vector<std::vector<int64_t>>{{2,3}, {1,1}}) {
    auto record = layout::BuildLayout({3,2}, {2,1}, 0, destination, 0, 6, 8, 0);
    auto decoded = descriptor::DecodedLayout{{record},6,8};
    const auto raw = descriptor::EncodeLayout(decoded);
    try {
      descriptor::DecodeAccumulationLayout(raw.data(), raw.size());
      throw std::runtime_error("unproven owners accepted");
    } catch (const std::invalid_argument& error) {
      Require(std::string(error.what()).find("cannot prove injective output owners") != std::string::npos,
              "wrong owner rejection");
    }
  }
}

void InvalidStaticPlans() {
  const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
  const auto destroyed = Tensor0StrideCudaAccumulationPreparedDestroyedCount();
  auto reject = [&](Harness& h, const std::string& message) {
    h.Upload();
    const auto before = h.Snapshot();
    auto state = native::InstantiateAccumulation({h.words.data(), h.words.size()},
                                                 {h.indices.data(), h.indices.size()});
    Require(!state.has_value(), "invalid static Accumulation plan created a state");
    Require(state.error().message().find(message) != std::string::npos,
            "unexpected static rejection: " + state.error().message());
    Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() == created &&
            Tensor0StrideCudaAccumulationPreparedDestroyedCount() == destroyed,
            "invalid static descriptor constructed an Accumulation state");
    Require(h.Snapshot() == before, "invalid static descriptor wrote storage");
  };
  for (const auto& words : std::vector<std::vector<int64_t>>{
      {1,8,8,2,1,0,0,2,1,1,1,0,8,1,1,1},
      {1,8,8,1,1,0,0,3,std::numeric_limits<int64_t>::max(),1}}) {
    Harness h; h.words = words; reject(h, "address");
  }
  {
    Harness h; h.words = {1,8,8,1,1,0,0,0,1,1}; h.indices = {1};
    reject(h, "record indices");
  }
  for (const auto& indices : std::vector<std::vector<int64_t>>{{-1}, {0,0}, {1,0}}) {
    Harness h; h.words = {1,8,8,2,1,0,0,0,1,1,1,0,0,0,1,1};
    h.indices = indices; reject(h, "record indices");
  }
  {
    Harness h; h.words = {1,8,8,1,2,0,0,4294967296LL,4294967296LL,0,0,0,0};
    reject(h, "element count overflows");
  }
}

void Rejections() {
  for (int kind = 0; kind < 5; ++kind) {
    Harness h;
    if (kind == 0) h.result.data = h.source.data;
    if (kind == 1) h.result.data = static_cast<float*>(h.source.data) + 1;
    if (kind == 2) h.coefficient.data = h.result.data;
    if (kind == 3) h.descriptor.data = h.result.data;
    if (kind == 4) h.result.data = static_cast<int64_t*>(h.descriptor.data) + 1;
    h.Upload(); auto state = Prepare(h.words, h.indices); h.Reject(state.get(), "overlap");
  }
  for (int kind = 0; kind < 6; ++kind) {
    Harness h; h.Upload();
    std::string message;
    switch (kind) {
      case 0: h.source.rank = 1; message = "rank-two"; break;
      case 1: --h.descriptor_count; message = "descriptor operand shape"; break;
      case 2: h.coefficient_count = 2; message = "batch count"; break;
      case 3: h.source.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 4: h.coefficient.dtype = XLA_FFI_DataType_S64; message = "coefficient dtype"; break;
      case 5: h.operand_count = 0; message = "record count"; break;
    }
    auto state = Prepare(h.words, h.indices); h.Reject(state.get(), message);
  }
  {
    Harness h; h.shape[0] = 2;
    h.words = {1,8,8,1,2,0,0,4294967295LL,4294967297LL,0,0,0,0};
    h.Upload(); auto state = Prepare(h.words, h.indices);
    const float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Reject(state.get(), "logical batch size overflows");
    h.operand_count = 0;
    auto no_coefficient = Prepare(h.words, {});
    h.Reject(no_coefficient.get(), "logical batch size overflows");
  }
  for (bool multiply : {false, true}) {
    Harness h;
    h.shape = {multiply ? 3 : 1, multiply ? std::numeric_limits<int64_t>::max() : std::numeric_limits<ptrdiff_t>::max() / 4 + 1};
    h.words = {1,h.shape[1],h.shape[1],0};
    h.indices.clear(); h.operand_count = 0;
    h.Upload(); auto state = Prepare(h.words, h.indices);
    h.Reject(state.get(), multiply ? "batch storage size overflows" : "buffer size exceeds addressable storage");
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
  auto state = Prepare(h.words, h.indices);
  auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr;
  Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), 8 * sizeof(float)));
  Produce<<<1,8,0,h.stream>>>(static_cast<float*>(h.source.data), static_cast<float*>(h.coefficient.data));
  const auto error = h.Invoke(state.get()); Require(!error.failure(), error.message());
  const auto destroyed = Tensor0StrideCudaAccumulationPreparedDestroyedCount();
  state.reset();
  Require(Tensor0StrideCudaAccumulationPreparedDestroyedCount() == destroyed + 1,
          "Accumulation state not destroyed before stream synchronization");
  Consume<<<1,8,0,h.stream>>>(static_cast<float*>(h.result.data), observed);
  Cuda(cudaMemcpyAsync(host, observed, 8 * sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  const std::array<float,8> expected{4,7,10,13,1,1,1,1};
  Require(std::equal(expected.begin(), expected.end(), host), "stream dependency failure");
  Cuda(cudaFreeHost(host));
}

void Successes() {
  {
    Harness h;
    h.words = {1,8,8,1,2,0,0,1,4096,0,0,1,0};
    h.Upload();
    auto state = Prepare(h.words, h.indices);
    Require(state->capacity == 4, "parallel fiber scratch capacity");
    h.scratch_shape[1] = state->capacity;
    h.Expect(state.get(), {8192,0,0,0,0,0,0,0});
    float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Expect(state.get(), {0,0,0,0,0,0,0,0});
    h.scratch_shape[1] = 3;
    h.Reject(state.get(), "scratch capacity");
    h.scratch_shape[1] = 4;
    h.scratch.data = h.result.data;
    h.Reject(state.get(), "overlap");
    h.scratch.data = h.coefficient.data;
    h.Reject(state.get(), "overlap");
  }
  { Harness h; h.Upload(); auto state = Prepare(h.words, h.indices); h.Expect(state.get(), {2,4,6,8,0,0,0,0}); }
  { Harness h; h.words = {1,8,8,1,1,0,0,0,1,1}; h.Upload();
    auto state = Prepare(h.words, h.indices); h.Expect(state.get(), {}); }
  { Harness h; h.indices.clear(); h.operand_count = 0; h.Upload();
    auto state = Prepare(h.words, h.indices);
    h.operand_count = 1;
    h.Reject(state.get(), "record count");
    h.operand_count = 0;
    h.Expect(state.get(), {1,2,3,4,0,0,0,0}); }
  {
    Harness h; h.words = {1,8,8,1,1,0,0,0,1,1}; h.Upload();
    auto state = Prepare(h.words, h.indices);
    const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
    h.coefficient.dtype = XLA_FFI_DataType_S64;
    h.Reject(state.get(), "coefficient dtype");
    h.coefficient.dtype = XLA_FFI_DataType_F32;
    std::array<int64_t,2> invalid_shape{1,1};
    h.coefficient.rank = 2; h.coefficient.dims = invalid_shape.data();
    h.Reject(state.get(), "scalar");
    h.coefficient.rank = 1; h.coefficient.dims = &h.coefficient_count;
    h.coefficient.data = h.result.data;
    h.Reject(state.get(), "overlap");
    h.coefficient.data = h.arena + 384;
    h.Expect(state.get(), {});
    Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() == created,
            "empty-record rejected calls recreated state");
  }
  {
    Harness h; h.shape[0] = 0; h.coefficient_count = 0; h.Upload();
    auto state = Prepare(h.words, h.indices);
    const auto before = h.Snapshot(); const auto error = h.Invoke(state.get());
    Require(!error.failure(), error.message()); Require(h.Snapshot() == before, "zero batch write");
  }
  {
    // UINT64_MAX contributions with zero coefficient must not read or iterate.
    Harness h; h.words = {1,8,8,1,2,0,0,4294967295LL,4294967297LL,0,0,0,0};
    h.Upload(); float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    auto state = Prepare(h.words, h.indices); h.Expect(state.get(), {});
  }
}

void RawOwnedFiberAddresses() {
  // Unnormalized axes: only the first output stride selects an owner;
  // negative source strides and zero-stride singleton axes remain visible.
  Harness h;
  h.words = {1,12,8,1, 4,4,1, 2,3,1,2, 6,-2,INT64_MIN,1, 1,0,0,0};
  h.shape = {2,12};
  std::array<int64_t,2> output_shape{2,8};
  h.result.dims = output_shape.data();
  h.coefficient_count = 2;
  h.Upload();
  auto state = Prepare(h.words, h.indices);
  Require(state->decoded.records[0].shape == std::vector<uint64_t>({2,3,1,2}) &&
          state->schedules[0].owners == 2 &&
          state->schedules[0].contributions == 6,
          "raw signed owner/fiber axes lost or misclassified");
  std::array<float,24> input{};
  for (int i = 0; i < 24; ++i) input[i] = float(i + 1);
  const std::array<float,2> factors{2,3};
  Cuda(cudaMemcpyAsync(h.source.data, input.data(), sizeof(input), cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.coefficient.data, factors.data(), sizeof(factors), cudaMemcpyHostToDevice, h.stream));
  const auto error = h.Invoke(state.get()); Require(!error.failure(), error.message());
  std::array<float,16> actual{}, expected{};
  Cuda(cudaMemcpyAsync(actual.data(), h.result.data, sizeof(actual), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  for (int batch = 0; batch < 2; ++batch) {
    for (int map = 0; map < 2; ++map) {
      float value = 0;
      for (int red = 0; red < 3; ++red) {
        for (int inner = 0; inner < 2; ++inner) {
          value += factors[batch] * input[batch * 12 + 4 + 6 * map - 2 * red + inner];
        }
      }
      expected[batch * 8 + 1 + map] = value;
    }
  }
  Require(actual == expected, "raw signed owned-fiber address or coefficient mismatch");
}

void RejectRawCollisions() {
  for (const auto& destination : std::vector<std::array<int64_t, 2>>{{1,1}, {3,2}}) {
    Harness h;
    h.words = {1,8,10,1, 2,3,2, 2,3, -3,1, destination[0],destination[1]};
    h.shape[0] = 0;
    h.Upload();
    const auto before = h.Snapshot();
    auto state = native::InstantiateAccumulation({h.words.data(), h.words.size()},
        {h.indices.data(), h.indices.size()});
    Require(!state.has_value() &&
        state.error().message().find("cannot prove injective output owners") != std::string::npos,
        "raw colliding or unproven owners accepted with empty batch");
    Require(h.Snapshot() == before, "invalid static owner selection wrote storage");
  }
}

void ReuseCoefficients() {
  Harness h;
  h.words = {1,8,8,1,0,1,1};
  h.Upload();
  auto state = Prepare(h.words, h.indices);
  const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
  std::array<float,24> input{};
  for (int i = 0; i < 24; ++i) input[i] = float(i + 1);
  Cuda(cudaMemcpyAsync(h.source.data, input.data(), sizeof(input), cudaMemcpyHostToDevice, h.stream));
  for (int64_t batches : {1, 3, 0, 2}) {
    h.shape[0] = batches;
    for (bool integer : {false, true}) {
      h.coefficient.dtype = integer ? XLA_FFI_DataType_S32 : XLA_FFI_DataType_F32;
      for (int form = 0; form < 3; ++form) {
        h.coefficient.rank = form == 0 ? 0 : 1;
        h.coefficient_count = form == 2 ? batches : 1;
        for (int coefficient : {0, 2, 4}) {
          std::array<int32_t,3> ints{};
          std::array<float,3> floats{};
          for (int b = 0; b < std::max<int64_t>(batches, 1); ++b) {
            ints[b] = coefficient + (form == 2 ? b : 0);
            floats[b] = static_cast<float>(ints[b]);
          }
          Cuda(cudaMemcpyAsync(h.coefficient.data, integer ? static_cast<const void*>(ints.data()) : floats.data(),
                               (form == 2 ? std::max<int64_t>(batches, 1) : 1) * 4,
                               cudaMemcpyHostToDevice, h.stream));
          const auto before = h.Snapshot();
          const auto error = h.Invoke(state.get());
          Require(!error.failure(), error.message());
          const auto after = h.Snapshot();
          if (batches == 0) {
            Require(after == before, "zero-batch reused Accumulation wrote storage");
            continue;
          }
          for (int b = 0; b < batches; ++b) {
            for (int i = 0; i < 8; ++i) {
              float actual;
              std::memcpy(&actual, after.data() + 256 + 4 * (b * 8 + i), 4);
              Require(actual == (i == 1 ? float((b * 8 + 2) * ints[form == 2 ? b : 0]) : 0.f),
                      "reused Accumulation coefficient mismatch");
            }
          }
        }
      }
    }
  }
  h.shape[0] = h.coefficient_count = 1;
  h.coefficient.rank = 1;
  h.coefficient.dtype = XLA_FFI_DataType_F32;
  h.result.data = h.source.data;
  h.Reject(state.get(), "overlap");
  h.result.data = h.arena + 256;
  h.coefficient.dtype = XLA_FFI_DataType_S64;
  h.Reject(state.get(), "coefficient dtype");
  h.coefficient.dtype = XLA_FFI_DataType_F32;
  h.coefficient_count = 2;
  h.Reject(state.get(), "batch count");
  h.coefficient_count = 1;
  h.operand_count = 0;
  h.Reject(state.get(), "record count");
  h.operand_count = 1;
  const float factor = 3;
  Cuda(cudaMemcpyAsync(h.coefficient.data, &factor, 4, cudaMemcpyHostToDevice, h.stream));
  h.Expect(state.get(), {0,6,0,0,0,0,0,0});
  Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() == created,
          "execute unexpectedly created Accumulation state");
}

// Exercise a cached record-to-operand map even after both host arrays expire.
void SparseMapReuse() {
  std::unique_ptr<State> state;
  {
    std::vector<int64_t> words{1,8,8,3,0,0,0,1,0,0,0,1,1,0,0,0};
    std::vector<int64_t> records{0,2};
    state = Prepare(words, records);
    std::fill(words.begin(), words.end(), -99);
    std::fill(records.begin(), records.end(), -99);
  }
  Harness h;
  h.words = {1,8,8,3,0,0,0,1,0,0,0,1,1,0,0,0};
  h.indices = {0,2}; h.Upload();
  h.operand_count = 2;
  const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
  for (float value : {2.f, 4.f}) {
    Harness another;
    another.words = h.words; another.indices = h.indices; another.Upload();
    another.operand_count = 2;
    Harness& target = value == 2.f ? h : another;
    const float other = value + 3;
    Cuda(cudaMemcpyAsync(target.coefficient.data, &value, sizeof(value), cudaMemcpyHostToDevice, target.stream));
    Cuda(cudaMemcpyAsync(target.second.data, &other, sizeof(other), cudaMemcpyHostToDevice, target.stream));
    const auto error = target.Invoke(state.get());
    Require(!error.failure(), error.message());
    std::array<float,8> actual{};
    Cuda(cudaMemcpyAsync(actual.data(), target.result.data, sizeof(actual), cudaMemcpyDeviceToHost, target.stream));
    Cuda(cudaStreamSynchronize(target.stream));
    Require(actual == std::array<float,8>{value + other,0,0,0,0,0,0,0},
            "sparse coefficient map or overlapping records lost");
  }
  Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() == created,
          "execute recreated Accumulation state");
}
}  // namespace

int main() {
  int devices = 0;
  const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && devices == 0)) {
    std::cout << "CUDA device unavailable\n"; return 77;
  }
  try {
    Cuda(status);
    const auto created = Tensor0StrideCudaAccumulationPreparedCreatedCount();
    const auto destroyed = Tensor0StrideCudaAccumulationPreparedDestroyedCount();
    Classifier(); InvalidStaticPlans(); Rejections(); Successes(); RawOwnedFiberAddresses(); RejectRawCollisions(); ReuseCoefficients(); SparseMapReuse(); StreamDependency();
    Require(Tensor0StrideCudaAccumulationPreparedCreatedCount() - created ==
            Tensor0StrideCudaAccumulationPreparedDestroyedCount() - destroyed,
            "Accumulation prepared state leaked");
    std::cout << "CUDA Accumulation contract passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
