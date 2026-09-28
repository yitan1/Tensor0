// Device-backed prepared Dot validation, reuse, metadata, and stream contracts.
#include "cuda/dot.cu"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>

struct XLA_FFI_Error { std::string message; };
namespace native = tensor0::stride::cuda::dot;
namespace layout = tensor0::stride::layout;
namespace descriptor = tensor0::stride::descriptor;
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
const std::vector<int64_t> kWords{1, 8, 8, 1, 1, 0, 0, 8, 1, 1};
using State = native::DotPreparedState;
std::unique_ptr<State> Prepare(const std::vector<int64_t>& words, int64_t conjugate = 0) {
  const auto created = Tensor0StrideCudaDotPreparedCreatedCount();
  auto state = native::InstantiateDot({words.data(), words.size()}, conjugate);
  Require(state.has_value(), "valid Dot plan rejected");
  Require(Tensor0StrideCudaDotPreparedCreatedCount() == created + 1,
          "instantiate did not create exactly one Dot state");
  return std::move(*state);
}
struct Context { State* state; cudaStream_t stream; };
XLA_FFI_Api Api() {
  XLA_FFI_Api api{};
  api.struct_size = XLA_FFI_Api_STRUCT_SIZE;
  api.api_version = {XLA_FFI_Api_Version_STRUCT_SIZE, nullptr,
                     XLA_FFI_API_MAJOR, XLA_FFI_API_MINOR};
  api.XLA_FFI_Error_Create = [](XLA_FFI_Error_Create_Args* args) {
    return new XLA_FFI_Error{args->message};
  };
  api.XLA_FFI_Error_GetMessage = [](XLA_FFI_Error_GetMessage_Args* args) {
    args->message = args->error->message.c_str();
  };
  api.XLA_FFI_Error_Destroy = [](XLA_FFI_Error_Destroy_Args* args) { delete args->error; };
  api.XLA_FFI_State_Get = [](XLA_FFI_State_Get_Args* args) -> XLA_FFI_Error* {
    args->state = reinterpret_cast<Context*>(args->ctx)->state;
    return nullptr;
  };
  api.XLA_FFI_Stream_Get = [](XLA_FFI_Stream_Get_Args* args) -> XLA_FFI_Error* {
    args->stream = reinterpret_cast<Context*>(args->ctx)->stream;
    return nullptr;
  };
  return api;
}
struct Harness {
  static constexpr size_t bytes = 4096;
  unsigned char* arena = nullptr;
  cudaStream_t stream = nullptr;
  std::array<int64_t, 2> shape{1, 8};
  int64_t batches = 1, descriptor_count = 0;
  std::array<int64_t, 2> scratch_shape{1, 0};
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
    scratch = Buffer(2048, XLA_FFI_DataType_F32, 2, scratch_shape.data());
    descriptor = Buffer(512, XLA_FFI_DataType_S64, 1, &descriptor_count);
    std::array<float, bytes / sizeof(float)> initial{};
    initial.fill(99);
    for (int i = 0; i < 8; ++i) { initial[i] = float(i + 1); initial[32 + i] = 2; }
    Cuda(cudaMemcpyAsync(arena, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    Cuda(cudaStreamSynchronize(stream));
  }
  ~Harness() { cudaStreamDestroy(stream); cudaFree(arena); }
  void Upload(const std::vector<int64_t>& words) {
    const auto packed = layout::PackOwnerFiber(
        descriptor::DecodeAddressLayout(words.data(), words.size()), 2);
    descriptor_count = packed.words.size();
    scratch_shape = {batches, static_cast<int64_t>(layout::DotScratchCapacity(packed.schedules))};
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
    return native::Dot<ffi::F32, float>(state, ffi::AnyBuffer(&left), ffi::AnyBuffer(&right),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::BufferR1<ffi::F32>(&result), ffi::BufferR2<ffi::F32>(&scratch), stream);
  }
  std::string BoundInvoke(State* state) {
    auto api = Api();
    State::id.type_id = 1;
    Context context{state, stream};
    XLA_FFI_ArgType arg_types[]{XLA_FFI_ArgType_BUFFER, XLA_FFI_ArgType_BUFFER,
                                XLA_FFI_ArgType_BUFFER};
    void* args[]{&left, &right, &descriptor};
    XLA_FFI_RetType ret_types[]{XLA_FFI_RetType_BUFFER, XLA_FFI_RetType_BUFFER};
    void* rets[]{&result, &scratch};
    XLA_FFI_CallFrame frame{};
    frame.struct_size = XLA_FFI_CallFrame_STRUCT_SIZE;
    frame.api = &api;
    frame.ctx = reinterpret_cast<XLA_FFI_ExecutionContext*>(&context);
    frame.stage = XLA_FFI_ExecutionStage_EXECUTE;
    frame.args = {XLA_FFI_Args_STRUCT_SIZE, nullptr, 3, arg_types, args};
    frame.rets = {XLA_FFI_Rets_STRUCT_SIZE, nullptr, 2, ret_types, rets};
    frame.attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr, 0, nullptr, nullptr, nullptr};
    auto* error = Tensor0StrideCudaDotF32V1(&frame);
    if (!error) return {};
    const auto message = error->message;
    delete error;
    return message;
  }
  void Reject(State* state, const std::string& message, bool bound = false) {
    const auto before = Snapshot();
    std::string actual;
    if (bound) actual = BoundInvoke(state);
    else {
      const auto error = Invoke(state);
      Require(error.failure(), "expected Dot rejection: " + message);
      actual = error.message();
    }
    Require(!actual.empty() && actual.find(message) != std::string::npos,
            "unexpected Dot rejection: " + actual);
    Require(Snapshot() == before, "rejected Dot changed result or other storage");
  }
  void CheckResults(std::initializer_list<float> expected) {
    Require(static_cast<int64_t>(expected.size()) == batches, "wrong Dot expected batch count");
    std::vector<float> actual(expected.size());
    Cuda(cudaMemcpyAsync(actual.data(), result.data, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    Require(std::equal(actual.begin(), actual.end(), expected.begin()), "wrong Dot batch output");
  }
  void Expect(State* state, float expected) {
    const auto error = Invoke(state); Require(!error.failure(), error.message());
    CheckResults({expected});
  }
};

void InvalidStaticPlans() {
  const std::vector<std::vector<int64_t>> invalid{
      {}, {2, 8, 8, 0}, {1, -1, 8, 0}, {1, 8, 8, 1},
      {1, 8, 8, 1, 1, 0, 0, 8, 1},
      {1, 8, 8, 1, 1, 0, 0, 9, 1, 1},
      {1, 8, 8, 1, 1, 0, 0, 8, INT64_MAX, 1},
      // 3 * INT64_MAX exceeds uint64_t; offsets and strides are otherwise valid.
      {1, 1, 1, 1, 2, 0, 0, 3, INT64_MAX, 0, 0, 0, 0},
      // First record is valid, second has an out-of-bounds source offset.
      {1, 8, 8, 2, 1, 0, 0, 8, 1, 1, 1, 8, 0, 1, 0, 0}};
  const auto created = Tensor0StrideCudaDotPreparedCreatedCount();
  const auto destroyed = Tensor0StrideCudaDotPreparedDestroyedCount();
  for (const auto& words : invalid) {
    auto state = native::InstantiateDot({words.data(), words.size()}, 0);
    Require(!state.has_value(), "invalid static Dot descriptor created a state");
    Require(Tensor0StrideCudaDotPreparedCreatedCount() == created &&
            Tensor0StrideCudaDotPreparedDestroyedCount() == destroyed,
            "invalid static Dot descriptor constructed then discarded a state");
  }
  auto conjugate = native::InstantiateDot({invalid.front().data(), invalid.front().size()}, 2);
  Require(!conjugate.has_value() && conjugate.error().message().find("zero or one") != std::string::npos,
          "conjugate must be checked before decoding the invalid descriptor");
  Require(Tensor0StrideCudaDotPreparedCreatedCount() == created &&
          Tensor0StrideCudaDotPreparedDestroyedCount() == destroyed,
          "invalid conjugate created a Dot state");
}

void Rejections(State* state) {
  for (int kind = 0; kind < 21; ++kind) {
    Harness h; h.Upload(kWords);
    std::string message;
    bool bound = false;
    switch (kind) {
      case 0: h.result.data = h.left.data; message = "overlap"; break;
      case 1: h.result.data = static_cast<float*>(h.right.data) + 1; message = "overlap"; break;
      case 2: h.descriptor.data = h.result.data; message = "overlap"; break;
      case 3: h.left.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 4: h.right.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 5: --h.descriptor_count; message = "descriptor operand shape"; break;
      case 6: h.descriptor_count = -1; message = "descriptor operand shape"; break;
      case 7: h.shape[1] = 7; message = "dimensions"; break;
      case 8: h.batches = 2; message = "dimensions"; break;
      case 9: h.shape[0] = -1; message = "nonnegative"; break;
      case 10: h.left.rank = 1; message = "rank-two"; break;
      case 11: h.result.rank = 2; bound = true; break;
      case 12: h.descriptor.dtype = XLA_FFI_DataType_F32; bound = true; break;
      case 13: h.right.rank = 1; message = "rank-two"; break;
      case 14: h.scratch_shape[1] = 2; message = "scratch capacity"; break;
      case 15: h.scratch_shape[0] = 2; message = "dimensions"; break;
      case 16: h.scratch.rank = 1; bound = true; break;
      case 17: h.scratch.data = h.left.data; message = "overlap"; break;
      case 18: h.scratch.data = h.result.data; message = "overlap"; break;
      case 19: h.scratch.data = h.descriptor.data; message = "overlap"; break;
      case 20: h.scratch.dtype = XLA_FFI_DataType_F64; bound = true; break;
    }
    h.Reject(state, message, bound);
  }
  { Harness h; h.Upload(kWords); h.shape[0] = h.batches = h.scratch_shape[0] = INT64_MAX;
    h.Reject(state, "logical batch size overflows"); }
  { Harness h; h.Upload(kWords); h.shape[0] = h.batches = h.scratch_shape[0] = INT64_MAX / 8;
    h.Reject(state, "addressable storage"); }
  Harness h; h.Upload(kWords); h.Expect(state, 72);  // Rejection never poisons the plan.
}

void ReuseAndBoundaries() {
  std::unique_ptr<State> state;
  {
    auto host = kWords;
    state = Prepare(host);
    std::fill(host.begin(), host.end(), -1);
  }  // Prepared metadata must own its words after the caller's storage expires.
  Require(state->word_count == layout::PackOwnerFiber(state->decoded, 2).words.size() && state->conjugate == 0 &&
          state->capacity == 1 &&
          state->decoded.records.size() == 1 && state->counts.size() == 1 &&
          state->counts[0] == 8 && state->schedules[0].owners == 1 && state->schedules[0].contributions == 8,
          "prepared Dot metadata differs from the validated descriptor");
  Harness h; h.Upload(kWords);
  h.Expect(state.get(), 72);
  h.right.data = h.left.data;
  h.Expect(state.get(), 204);  // Inputs may alias each other.
  h.right.data = h.arena + 128;
  for (int64_t batches : {0, 1, 2, 0, 1}) {
    h.shape[0] = h.batches = h.scratch_shape[0] = batches;
    const auto before = h.Snapshot();
    const auto error = h.Invoke(state.get()); Require(!error.failure(), error.message());
    if (batches == 0) Require(h.Snapshot() == before, "zero-batch Dot wrote storage");
    else if (batches == 1) h.CheckResults({72});
    else h.CheckResults({72, 78408});  // The second batch uses untouched 99-valued inputs.
  }
  Rejections(state.get());
  auto conjugated = Prepare(kWords, 1);
  Require(conjugated->conjugate == 1, "prepared Dot lost conjugate flag");
  h.Expect(conjugated.get(), 72);
  const auto created = Tensor0StrideCudaDotPreparedCreatedCount();
  Require(h.BoundInvoke(state.get()).empty(), "valid bound Dot execution failed");
  Cuda(cudaStreamSynchronize(h.stream));
  Require(Tensor0StrideCudaDotPreparedCreatedCount() == created,
          "execute unexpectedly instantiated another Dot state");

  // Rank-one record encoding: rank, source offset, destination offset,
  // extent, source stride, destination stride. Every address is within [0,8).
  const std::vector<int64_t> boundary{
      1,8,8,4,
      1,0,0,1023,0,0,
      2,0,0,0,2,0,0,0,0,  // Empty rank-two record advances cursor by nine words.
      1,1,0,1024,0,0,
      1,2,0,1025,0,0};
  auto boundary_state = Prepare(boundary);
  Require(boundary_state->counts.size() == 4 && boundary_state->counts[0] == 1023 &&
          boundary_state->counts[1] == 0 && boundary_state->counts[2] == 1024 &&
          boundary_state->counts[3] == 1025 && boundary_state->schedules.size() == 4 &&
          boundary_state->schedules[0].contributions == 1023 &&
          boundary_state->schedules[1].contributions == 0 &&
          boundary_state->schedules[2].contributions == 1024 &&
          boundary_state->schedules[3].contributions == 1025 && boundary_state->capacity == 2 &&
          boundary_state->word_count == layout::PackOwnerFiber(boundary_state->decoded, 2).words.size(),
          "Dot owner/fiber boundary metadata incorrect");
  Harness b; b.Upload(boundary);
  b.Expect(boundary_state.get(), 12292);  // Sequential owner/fiber sum over four records.
  const std::vector<int64_t> scalar{1,8,8,1,0,1,2};
  auto scalar_state = Prepare(scalar);
  Require(scalar_state->counts[0] == 1 && scalar_state->schedules[0].contributions == 1, "rank-zero Dot metadata incorrect");
  Harness r; r.Upload(scalar); r.Expect(scalar_state.get(), 4);
  const std::vector<int64_t> empty{1,8,8,2, 1,0,0,0,0,0, 1,0,0,0,0,0};
  auto empty_state = Prepare(empty);
  Require(empty_state->schedules[0].contributions == 0 && empty_state->counts[0] == 0 &&
          empty_state->counts[1] == 0, "all-empty Dot must have zero contributions");
  Harness e; e.Upload(empty); e.Expect(empty_state.get(), 0);
  const std::vector<int64_t> zero_fiber{1,8,8,1,1,0,0,0,1,1};
  auto zero_state = Prepare(zero_fiber);
  Harness z; z.Upload(zero_fiber); z.Expect(zero_state.get(), 0);
  const std::vector<int64_t> no_records{1,8,8,0};
  auto no_records_state = Prepare(no_records);
  Harness n; n.Upload(no_records);
  n.Expect(no_records_state.get(), 0);
  n.shape[0] = n.batches = n.scratch_shape[0] = INT64_MAX;
  n.Reject(no_records_state.get(), "batch storage size overflows");
  const std::vector<int64_t> no_storage{1,0,0,0};
  auto no_storage_state = Prepare(no_storage);
  Harness s; s.Upload(no_storage); s.shape[1] = 0;
  s.shape[0] = s.batches = s.scratch_shape[0] = INT64_MAX;
  s.Reject(no_storage_state.get(), "addressable storage");
  const std::vector<int64_t> large_source{1,INT64_MAX,8,0};
  auto large_source_state = Prepare(large_source);
  Harness l; l.Upload(large_source); l.shape[1] = INT64_MAX;
  l.Reject(large_source_state.get(), "dimensions");
}

void RawMultiaxisAddresses() {
  // Bypass lowering normalization: both axes must reach the device decoder.
  const std::vector<int64_t> words{1,8,8,1, 2,3,2, 2,3, -3,1, 0,-1};
  auto plain = Prepare(words);
  auto conjugated = Prepare(words, 1);
  Require(plain->decoded.records[0].shape.size() == 2 && plain->counts[0] == 6,
          "raw Dot record unexpectedly normalized");
  Harness h; h.Upload(words);
  h.shape[0] = h.batches = h.scratch_shape[0] = 2;
  std::array<float,16> left{}, right{};
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 8; ++i) {
      left[b * 8 + i] = float(1 + b * 8 + i);
      right[b * 8 + i] = float(2 + b * 3 + i);
    }
  }
  Cuda(cudaMemcpyAsync(h.left.data, left.data(), sizeof(left), cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.right.data, right.data(), sizeof(right), cudaMemcpyHostToDevice, h.stream));
  for (auto* state : {plain.get(), conjugated.get()}) {
    const auto error = h.Invoke(state); Require(!error.failure(), error.message());
    std::array<float,2> actual{};
    Cuda(cudaMemcpyAsync(actual.data(), h.result.data, sizeof(actual), cudaMemcpyDeviceToHost, h.stream));
    Cuda(cudaStreamSynchronize(h.stream));
    for (int b = 0; b < 2; ++b) {
      float expected = 0;
      for (int outer = 0; outer < 2; ++outer) {
        for (int inner = 0; inner < 3; ++inner) {
          expected += left[b * 8 + 3 - 3 * outer + inner] *
                      right[b * 8 + 2 - inner];
        }
      }
      Require(actual[b] == expected, "raw multiaxis Dot address or batch mismatch");
    }
  }
  using Complex = tensor0::stride::cuda::arithmetic::Complex<float>;
  h.left.dtype = h.right.dtype = h.result.dtype = h.scratch.dtype = XLA_FFI_DataType_C64;
  std::array<Complex,16> complex_left{}, complex_right{};
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 8; ++i) {
      complex_left[b * 8 + i] = {left[b * 8 + i], float(i - b)};
      complex_right[b * 8 + i] = {right[b * 8 + i], float(i + b)};
    }
  }
  Cuda(cudaMemcpyAsync(h.left.data, complex_left.data(), sizeof(complex_left), cudaMemcpyHostToDevice, h.stream));
  Cuda(cudaMemcpyAsync(h.right.data, complex_right.data(), sizeof(complex_right), cudaMemcpyHostToDevice, h.stream));
  const auto error = native::Dot<ffi::C64, Complex>(conjugated.get(), ffi::AnyBuffer(&h.left),
      ffi::AnyBuffer(&h.right), ffi::BufferR1<ffi::S64>(&h.descriptor),
      ffi::BufferR1<ffi::C64>(&h.result), ffi::BufferR2<ffi::C64>(&h.scratch), h.stream);
  Require(!error.failure(), error.message());
  std::array<Complex,2> actual{};
  Cuda(cudaMemcpyAsync(actual.data(), h.result.data, sizeof(actual), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));
  for (int b = 0; b < 2; ++b) {
    float real = 0, imag = 0;
    for (int outer = 0; outer < 2; ++outer) {
      for (int inner = 0; inner < 3; ++inner) {
        const auto a = complex_left[b * 8 + 3 - 3 * outer + inner];
        const auto r = complex_right[b * 8 + 2 - inner];
        real += a.real * r.real + a.imag * r.imag;
        imag += a.real * r.imag - a.imag * r.real;
      }
    }
    Require(actual[b].real == real && actual[b].imag == imag,
            "raw multiaxis conjugated complex Dot mismatch");
  }
}

void HugeCounts() {
  // 3 * 6148914691236517205 == UINT64_MAX; repeated address zero is valid.
  const std::vector<int64_t> huge{1,1,1,1,2,0,0,
      3,6148914691236517205LL,0,0,0,0};
  auto state = Prepare(huge);
  Require(state->counts.size() == 1 && state->counts[0] == UINT64_MAX &&
          state->schedules[0].owners == 1 &&
          state->schedules[0].contributions == UINT64_MAX,
          "UINT64_MAX count must remain a single owner/fiber schedule");
  Harness h; h.Upload(huge); h.shape[1] = 1;
  h.shape[0] = h.batches = h.scratch_shape[0] = 2;
  h.Reject(state.get(), "logical batch size overflows");
  // Never execute this huge fiber: its single-batch count is representable.

}

__global__ void Produce(float* left, float* right) {
  const auto start = clock64();
  while (clock64() - start < 1000000) {}
  left[threadIdx.x] = threadIdx.x + 1;
  right[threadIdx.x] = 3;
}
__global__ void Consume(const float* result, float* observed) { *observed = *result + 1; }
void StreamAndStateLifetime() {
  Harness h; h.Upload(kWords);
  auto state = Prepare(kWords);
  auto* observed = reinterpret_cast<float*>(h.arena + 1024);
  float* host = nullptr; Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), sizeof(float)));
  Produce<<<1,8,0,h.stream>>>(static_cast<float*>(h.left.data), static_cast<float*>(h.right.data));
  Cuda(cudaGetLastError());
  const auto error = h.Invoke(state.get()); Require(!error.failure(), error.message());
  const auto destroyed = Tensor0StrideCudaDotPreparedDestroyedCount();
  state.reset();  // Device work must not depend on this host state's lifetime.
  Require(Tensor0StrideCudaDotPreparedDestroyedCount() == destroyed + 1,
          "Dot host state not released before stream synchronization");
  Consume<<<1,1,0,h.stream>>>(static_cast<float*>(h.result.data), observed);
  Cuda(cudaGetLastError());
  Cuda(cudaMemcpyAsync(host, observed, sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream)); Require(*host == 109, "stream dependency failure");
  Cuda(cudaFreeHost(host));
}
}  // namespace
int main() {
  int devices = 0; const auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      (status == cudaSuccess && devices == 0)) { std::cout << "CUDA device unavailable\n"; return 77; }
  try {
    Cuda(status);
    const auto created = Tensor0StrideCudaDotPreparedCreatedCount();
    const auto destroyed = Tensor0StrideCudaDotPreparedDestroyedCount();
    InvalidStaticPlans(); ReuseAndBoundaries(); RawMultiaxisAddresses(); HugeCounts(); StreamAndStateLifetime();
    Require(Tensor0StrideCudaDotPreparedCreatedCount() - created ==
            Tensor0StrideCudaDotPreparedDestroyedCount() - destroyed,
            "Dot prepared state leaked");
    std::cout << "CUDA Dot contract passed\n"; return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
