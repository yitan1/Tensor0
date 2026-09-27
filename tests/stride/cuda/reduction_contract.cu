// Device-backed reduction validation and non-default-stream ordering.
#include "cuda/reduction.cu"

#include <array>
#include <cstring>
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
  int64_t operand_count = 1;
  std::array<int64_t, 2> scratch_shape{1, 0};
  XLA_FFI_Buffer source{}, result{}, scratch{}, coefficient{}, second{}, descriptor{};
  std::unique_ptr<tensor0::stride::cuda::reduction::ReductionPreparedState> state;
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
      std::vector<char> raw;
      for (auto word : words)
        for (unsigned shift = 0; shift < 64; shift += 8)
          raw.push_back(static_cast<char>((static_cast<uint64_t>(word) >> shift) & 255));
      metadata = tensor0::stride::layout::PackOwnerFiber(tensor0::stride::descriptor::DecodeReductionLayout(raw.data(), raw.size())).words;
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
  ffi::Error Prepare() {
    auto prepared = tensor0::stride::cuda::reduction::InstantiateReduction(
        {words.data(), words.size()}, {indices.data(), indices.size()});
    if (!prepared) return prepared.error();
    state = std::move(*prepared);
    scratch_shape[0] = shape[0];
    scratch_shape[1] = state->capacity;
    return ffi::Error::Success();
  }
  ffi::Error Invoke() {
    Require(state != nullptr, "invoke requires explicit prepared state");
    scratch_shape[0] = shape[0];
    XLA_FFI_ArgType types[2]{XLA_FFI_ArgType_BUFFER, XLA_FFI_ArgType_BUFFER};
    void* pointers[2]{&coefficient, &second};
    XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, operand_count, types, pointers};
    return tensor0::stride::cuda::reduction::Reduction<ffi::F32, float, float>(
        state.get(), ffi::AnyBuffer(&source),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::RemainingArgs(&args, 0),
        ffi::BufferR2<ffi::F32>(&result), ffi::BufferR2<ffi::F32>(&scratch), stream);
  }
  void Reject(const std::string& message) {
    if (!state) { const auto error = Prepare(); Require(!error.failure(), error.message()); }
    const auto before = Snapshot();
    const auto error = Invoke();
    Require(error.failure(), "expected rejection: " + message);
    Require(error.message().find(message) != std::string::npos, "unexpected rejection: " + error.message());
    Require(Snapshot() == before, "rejected invocation changed device storage");
  }
  void RejectStatic(const std::string& message) {
    const auto before = Snapshot();
    const auto created = Tensor0StrideCudaReductionPreparedCreatedCount();
    const auto error = Prepare();
    Require(error.failure() && error.message().find(message) != std::string::npos,
            "unexpected static rejection: " + error.message());
    Require(Tensor0StrideCudaReductionPreparedCreatedCount() == created, "invalid state created");
    Require(Snapshot() == before, "invalid static plan changed storage");
  }
  void Expect(std::array<float, 8> expected) {
    if (!state) { const auto error = Prepare(); Require(!error.failure(), error.message()); }
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
      case 5: h.coefficient_count = -1; message = "batch count"; break;
      case 6: h.coefficient.dtype = XLA_FFI_DataType_U8; message = "coefficient dtype"; break;
      case 7: h.shape[0] = -1; message = "nonnegative"; break;
      case 8: h.shape[1] = 9; message = "dimensions do not match"; break;
      case 9: h.shape[1] = -1; message = "nonnegative"; break;
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
    h.Upload(); if (kind == 11) h.Reject(message); else h.RejectStatic(message);
  }
  {
    // An invalid later record must fail before the earlier valid one writes.
    Harness h;
    h.words[3] = 2;
    h.words.insert(h.words.end(), {1, 0, 8, 1, 1, 1, 1, 0});
    h.Upload(); h.RejectStatic("address exceeds");
  }
  for (const std::vector<int64_t> indices : {std::vector<int64_t>{-1}, {1}, {0, 0}}) {
    Harness h; h.indices = indices; h.Upload(); h.RejectStatic("record indices");
  }
  for (bool trailing : {false, true}) {
    // Original-axis prefix overflow must precede even a trailing-word error.
    Harness h;
    h.words = {1,8,8,1,3,0,0,-1,2,0,0,0,0,1,2,1,0,1,0,1,0,1};
    if (trailing) { h.words[3] = 2; h.words.insert(h.words.end(),
        {1, 9, 0, 1, 1, 1, 1, 0}); h.words.push_back(0); }
    h.Upload(); h.RejectStatic("layout element count overflows");
  }
  for (bool multiply : {false, true}) {
    Harness h;
    h.shape = {multiply ? 3 : 1, multiply ? std::numeric_limits<int64_t>::max() : std::numeric_limits<ptrdiff_t>::max() / 4 + 1};
    h.words = {1,h.shape[1],h.shape[1],0}; h.indices.clear(); h.operand_count = 0;
    h.Upload(); h.Reject(multiply ? "batch storage size overflows" : "buffer size exceeds addressable storage");
  }
}

void RawInterleavedRoles() {
  // Raw rank-four descriptor: the singleton map axis has zero output stride,
  // but its explicit role must not turn it into a reduction axis.
  Harness h;
  h.words = {1,12,8,1, 4,4,1, 2,3,1,2, 6,-2,INT64_MIN,1,
             2,1,1,1, 1,99,0,99, 0,1,0,1};
  h.shape = {2,12};
  std::array<int64_t,2> output_shape{2,8};
  h.result.dims = output_shape.data();
  h.coefficient.dtype = XLA_FFI_DataType_S32;
  h.coefficient_count = 2;
  h.Upload();
  const auto prepared = h.Prepare(); Require(!prepared.failure(), prepared.message());
  const auto& record = h.state->decoded.records[0];
  Require(record.layout.shape == std::vector<uint64_t>({2,3,1,2}) &&
          record.reduction_axes == std::vector<bool>({false,true,false,true}) &&
          h.state->schedules[0].owners == 2 && h.state->schedules[0].contributions == 6,
          "raw interleaved reduction roles or singleton map axis lost");
  std::array<float,24> input{};
  for (int i = 0; i < 24; ++i) input[i] = float(i + 1);
  const std::array<int32_t,2> factors{2,3};
  Cuda(cudaMemcpyAsync(h.source.data, input.data(), sizeof(input), cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.coefficient.data, factors.data(), sizeof(factors), cudaMemcpyHostToDevice, h.stream));
  const auto error = h.Invoke(); Require(!error.failure(), error.message());
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
  Require(actual == expected, "raw interleaved reduction address, role or coefficient mismatch");
}

void DynamicReuse() {
  Harness h; h.Upload();
  const auto error = h.Prepare(); Require(!error.failure(), error.message());
  const auto created = Tensor0StrideCudaReductionPreparedCreatedCount();
  std::array<float,24> input{};
  for (int i = 0; i < 24; ++i) input[i] = i + 1;
  Cuda(cudaMemcpyAsync(h.source.data, input.data(), sizeof(input), cudaMemcpyHostToDevice, h.stream));
  for (int64_t batches : {1, 2, 0, 3}) {
    h.shape[0] = batches;
    for (bool integer : {false, true}) {
      h.coefficient.dtype = integer ? XLA_FFI_DataType_S32 : XLA_FFI_DataType_F32;
      for (int form = 0; form < 3; ++form) {
        h.coefficient.rank = form == 0 ? 0 : 1;
        h.coefficient_count = form == 2 ? batches : 1;
        for (int factor : {0, 1, 2}) {
          std::array<int32_t,3> ints{};
          std::array<float,3> floats{};
          for (int b = 0; b < std::max<int64_t>(batches, 1); ++b) {
            ints[b] = factor + (form == 2 ? b : 0);
            floats[b] = static_cast<float>(ints[b]);
          }
          Cuda(cudaMemcpyAsync(h.coefficient.data, integer ? static_cast<const void*>(ints.data()) : floats.data(),
              (form == 2 ? std::max<int64_t>(batches, 1) : 1) * 4, cudaMemcpyHostToDevice, h.stream));
          const auto before = h.Snapshot();
          const auto result = h.Invoke(); Require(!result.failure(), result.message());
          const auto after = h.Snapshot();
          if (batches == 0) { Require(after == before, "zero batch wrote storage"); continue; }
          for (int b = 0; b < batches; ++b) {
            for (int index = 0; index < 8; ++index) {
              float actual;
              std::memcpy(&actual, after.data() + 256 + 4 * (b * 8 + index), 4);
              const auto value = index < 2 ? float((16 * b + 4 + 2 * index) * ints[form == 2 ? b : 0]) : 0.f;
              Require(actual == value, "reused reduction coefficient or batch mapping mismatch");
            }
          }
        }
      }
    }
  }
  h.shape[0] = h.coefficient_count = 1;
  h.coefficient.rank = 1; h.coefficient.dtype = XLA_FFI_DataType_F32;
  h.source.data = h.arena + 768;
  Cuda(cudaMemcpyAsync(h.source.data, input.data(), 8 * sizeof(float), cudaMemcpyHostToDevice, h.stream));
  float factor = 2;
  Cuda(cudaMemcpyAsync(h.coefficient.data, &factor, sizeof(factor), cudaMemcpyHostToDevice, h.stream));
  h.Expect({8,12,0,0,0,0,0,0});
  Require(Tensor0StrideCudaReductionPreparedCreatedCount() == created, "dynamic reuse recreated host state");
}

void SparseEmptyMapping() {
  Harness h;
  const auto original = h.words;
  h.words = {1, 8, 8, 2, 1, 0, 0, 0, 1, 0, 1, 0};
  h.words.insert(h.words.end(), original.begin() + 4, original.end());
  h.indices = {0, 1}; h.operand_count = 2;
  h.Upload();
  const auto prepared = h.Prepare(); Require(!prepared.failure(), prepared.message());
  float first = 0, second = 2;
  Cuda(cudaMemcpyAsync(h.coefficient.data, &first, 4, cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.second.data, &second, 4, cudaMemcpyHostToDevice, h.stream));
  h.Expect({8,12,0,0,0,0,0,0});
  h.second.dtype = XLA_FFI_DataType_S64; h.Reject("coefficient dtype");
  h.second.dtype = XLA_FFI_DataType_F32;
  h.coefficient.dtype = XLA_FFI_DataType_S64; h.Reject("coefficient dtype");
  h.coefficient.dtype = XLA_FFI_DataType_F32;
  h.coefficient_count = 2; h.Reject("batch count");
  h.coefficient_count = 1;
  h.coefficient.data = h.result.data; h.Reject("overlap");
  h.coefficient.data = h.arena + 384;
  h.Expect({8,12,0,0,0,0,0,0});
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
  const auto prepared = h.Prepare(); Require(!prepared.failure(), prepared.message());
  Produce<<<1,8,0,h.stream>>>(static_cast<float*>(h.source.data), static_cast<float*>(h.coefficient.data));
  const auto error = h.Invoke(); Require(!error.failure(), error.message());
  h.state.reset();
  Consume<<<1,8,0,h.stream>>>(static_cast<float*>(h.result.data), observed);
  Cuda(cudaMemcpyAsync(host, observed, 8 * sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  const std::array<float,8> expected{13,19,1,1,1,1,1,1};
  Require(std::equal(expected.begin(), expected.end(), host), "stream dependency failure");
  Cuda(cudaFreeHost(host));
}

void Successes() {
  {
    // This plan selects the parallel strategy but a zero coefficient skips
    // both device stages without reading scratch or the enormous fiber.
    Harness h;
    h.words = {1,8,8,1,2,0,0,1,4096,0,0,1,1,1,0,0,1};
    h.Upload();
    const auto prepared = h.Prepare(); Require(!prepared.failure(), prepared.message());
    Require(h.state->capacity == 4, "parallel fiber scratch capacity");
    float one = 1;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &one, sizeof(one), cudaMemcpyHostToDevice, h.stream));
    h.Expect({4096,0,0,0,0,0,0,0});
    float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Expect({0,0,0,0,0,0,0,0});
    h.scratch_shape[1] = 3;
    h.Reject("scratch capacity");
    h.scratch_shape[1] = 4;
    h.scratch.data = h.result.data;
    h.Reject("overlap");
    h.scratch.data = h.source.data;
    h.Reject("overlap");
  }
  {
    Harness h; h.Upload();
    const auto error = h.Prepare(); Require(!error.failure(), error.message());
    const auto created = Tensor0StrideCudaReductionPreparedCreatedCount();
    h.coefficient_count = 2; h.Reject("batch count");
    h.coefficient_count = 1;
    h.operand_count = 0; h.Reject("record count"); h.operand_count = 1;
    h.Expect({8,12,0,0,0,0,0,0});
    h.words.assign(100, 99);  // Host attributes may expire after instantiate.
    h.Expect({8,12,0,0,0,0,0,0});
    Require(Tensor0StrideCudaReductionPreparedCreatedCount() == created, "reuse created another state");
  }
  { Harness h; h.words = {1,8,8,1,1,0,0,0,1,1,99,1}; h.Upload(); h.Expect({}); }
  {
    Harness h; h.shape[0] = 0; h.coefficient_count = 0; h.Upload();
    const auto prepared = h.Prepare(); Require(!prepared.failure(), prepared.message());
    const auto before = h.Snapshot(); const auto error = h.Invoke();
    Require(!error.failure(), error.message()); Require(h.Snapshot() == before, "zero batch write");
  }
  {
    // UINT64_MAX contributions with zero coefficient must not read or iterate.
    Harness h; h.words = {1,8,8,1,1,0,0,-1,0,1,99,1};
    h.Upload(); float zero = 0;
    Cuda(cudaMemcpyAsync(h.coefficient.data, &zero, sizeof(zero), cudaMemcpyHostToDevice, h.stream));
    h.Expect({});
    h.shape[0] = 2;
    h.Reject("logical batch size overflows");
    h.shape[0] = 1;
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
    Cuda(status);
    const auto created = Tensor0StrideCudaReductionPreparedCreatedCount();
    const auto destroyed = Tensor0StrideCudaReductionPreparedDestroyedCount();
    Rejections(); Successes(); RawInterleavedRoles(); DynamicReuse(); SparseEmptyMapping(); StreamDependency();
    Require(Tensor0StrideCudaReductionPreparedCreatedCount() - created ==
            Tensor0StrideCudaReductionPreparedDestroyedCount() - destroyed,
            "unbalanced reduction host states");
    std::cout << "CUDA Reduction contract passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
