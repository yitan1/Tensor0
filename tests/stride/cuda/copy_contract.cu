// Device-backed prepared Copy lifetime, validation, and stream-order contracts.
#include "cuda/copy.cu"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>

// Only the error/state/stream services used by the real execute handler are mocked.
struct XLA_FFI_Error { std::string message; };
namespace native = tensor0::stride::cuda;
namespace layout = tensor0::stride::layout;
namespace descriptor = tensor0::stride::descriptor;
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void Cuda(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
const std::vector<int64_t> kWords{1, 8, 8, 1, 1, 1, 2, 3, 2, 2};
using State = native::CopyPreparedState;
std::unique_ptr<State> Prepare(const std::vector<int64_t>& words) {
  const auto created = Tensor0StrideCudaCopyPreparedCreatedCount();
  auto state = native::InstantiateCopy({words.data(), words.size()});
  Require(state.has_value(), "valid Copy plan rejected");
  Require(Tensor0StrideCudaCopyPreparedCreatedCount() == created + 1,
          "instantiate did not create exactly one Copy state");
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
  std::array<int64_t, 2> source_shape{1, 8}, result_shape{1, 8};
  int64_t descriptor_count = kWords.size();
  XLA_FFI_Buffer source{}, result{}, descriptor{};
  XLA_FFI_Buffer Buffer(size_t offset, XLA_FFI_DataType dtype, int64_t rank, int64_t* dims) {
    return {XLA_FFI_Buffer_STRUCT_SIZE, nullptr, dtype, arena + offset, rank, dims};
  }
  Harness() {
    Cuda(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    source = Buffer(0, XLA_FFI_DataType_F32, 2, source_shape.data());
    result = Buffer(256, XLA_FFI_DataType_F32, 2, result_shape.data());
    descriptor = Buffer(1024, XLA_FFI_DataType_S64, 1, &descriptor_count);
    std::array<float, bytes / sizeof(float)> initial;
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = float(i + 1);
    Cuda(cudaMemcpyAsync(arena, initial.data(), bytes, cudaMemcpyHostToDevice, stream));
    Upload(kWords);
  }
  ~Harness() { cudaStreamDestroy(stream); cudaFree(arena); }
  void Upload(const std::vector<int64_t>& words) {
    const auto packed = layout::PackOwnerFiber(
        descriptor::DecodeLayout(words.data(), words.size()), 0);
    descriptor_count = packed.words.size();
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
    return native::Copy<ffi::F32, 4>(state, ffi::AnyBuffer(&source),
        ffi::BufferR1<ffi::S64>(&descriptor), ffi::BufferR2<ffi::F32>(&result), stream);
  }
  // Typed buffer rank/dtype validation belongs to FFI decoding, not direct calls.
  std::string BoundInvoke(State* state) {
    auto api = Api();
    State::id.type_id = 1;
    Context context{state, stream};
    XLA_FFI_ArgType arg_types[]{XLA_FFI_ArgType_BUFFER, XLA_FFI_ArgType_BUFFER};
    void* args[]{&source, &descriptor};
    XLA_FFI_RetType ret_types[]{XLA_FFI_RetType_BUFFER};
    void* rets[]{&result};
    XLA_FFI_CallFrame frame{};
    frame.struct_size = XLA_FFI_CallFrame_STRUCT_SIZE;
    frame.api = &api;
    frame.ctx = reinterpret_cast<XLA_FFI_ExecutionContext*>(&context);
    frame.stage = XLA_FFI_ExecutionStage_EXECUTE;
    frame.args = {XLA_FFI_Args_STRUCT_SIZE, nullptr, 2, arg_types, args};
    frame.rets = {XLA_FFI_Rets_STRUCT_SIZE, nullptr, 1, ret_types, rets};
    frame.attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr, 0, nullptr, nullptr, nullptr};
    auto* error = Tensor0StrideCudaCopyF32V1(&frame);
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
      auto error = Invoke(state);
      Require(error.failure(), "expected Copy rejection: " + message);
      actual = error.message();
    }
    Require(!actual.empty() && actual.find(message) != std::string::npos,
            "unexpected Copy rejection: " + actual);
    Require(Snapshot() == before, "rejected Copy submitted device writes");
  }
  void Accept(State* state) {
    const auto before = Snapshot();
    const auto error = Invoke(state);
    Require(!error.failure(), error.message());
    const auto after = Snapshot();
    if (source_shape[0] == 0) {
      Require(before == after, "zero-batch Copy wrote storage");
      return;
    }
    const size_t src_offset = static_cast<unsigned char*>(source.data) - arena;
    const size_t dst_offset = static_cast<unsigned char*>(result.data) - arena;
    for (int64_t batch = 0; batch < source_shape[0]; ++batch) {
      for (int i = 0; i < 8; ++i) {
        float expected = 0, actual;
        if (i == 2 || i == 4 || i == 6)
          std::memcpy(&expected, before.data() + src_offset + (batch * 8 + i - 1) * 4, 4);
        std::memcpy(&actual, after.data() + dst_offset + (batch * 8 + i) * 4, 4);
        Require(actual == expected, "Copy mapping or zero-filled hole mismatch");
      }
    }
    Require(std::equal(before.begin() + 1024, before.begin() + 1024 + descriptor_count * 8,
                       after.begin() + 1024), "Copy changed caller-owned device descriptor");
  }
};

void ReuseAndHostLifetime() {
  std::unique_ptr<State> state;
  {
    auto words = kWords;
    state = Prepare(words);
    std::fill(words.begin(), words.end(), -123);
    Harness h;
    h.Accept(state.get());  // Mutated host words are not reread.
  }
  // The original host allocation and all first-call device buffers have expired.
  const auto created = Tensor0StrideCudaCopyPreparedCreatedCount();
  Harness h;
  h.source.data = h.arena + 128;
  h.result.data = h.arena + 512;
  for (int64_t batches : {3, 0, 2, 1}) {
    h.source_shape[0] = h.result_shape[0] = batches;
    h.Accept(state.get());
  }
  --h.descriptor_count;
  h.Reject(state.get(), "descriptor operand shape");
  ++h.descriptor_count;
  h.Accept(state.get());
  Require(h.BoundInvoke(state.get()).empty(), "valid FFI execute failed");
  Cuda(cudaStreamSynchronize(h.stream));
  Require(Tensor0StrideCudaCopyPreparedCreatedCount() == created,
          "execute unexpectedly created another prepared state");
}

void Rejections() {
  auto state = Prepare(kWords);
  for (int kind = 0; kind < 16; ++kind) {
    Harness h;
    std::string message;
    bool bound = false;
    switch (kind) {
      case 0: h.source.dtype = XLA_FFI_DataType_F64; message = "same-dtype"; break;
      case 1: h.source.rank = 1; message = "rank-two"; break;
      case 2: h.source_shape[0] = -1; message = "nonnegative"; break;
      case 3: h.result_shape[1] = -1; message = "nonnegative"; break;
      case 4: h.source_shape[1] = 7; message = "dimensions"; break;
      case 5: h.result_shape[0] = 2; message = "dimensions"; break;
      case 6: --h.descriptor_count; message = "descriptor operand shape"; break;
      case 7: h.descriptor_count = -1; message = "descriptor operand shape"; break;
      case 8: h.result.data = h.source.data; message = "overlap"; break;
      case 9: h.result.data = h.arena + 4; message = "overlap"; break;
      case 10: h.result.data = h.descriptor.data; message = "overlap"; break;
      case 11: h.result.data = h.arena + 1032; message = "overlap"; break;
      case 12: h.result.rank = 1; bound = true; break;
      case 13: h.descriptor.rank = 2; bound = true; break;
      case 14: h.result.dtype = XLA_FFI_DataType_F64; bound = true; break;
      case 15: h.descriptor.dtype = XLA_FFI_DataType_F32; bound = true; break;
    }
    h.Reject(state.get(), message, bound);
  }
  for (bool multiply : {false, true}) {
    Harness h;
    h.source_shape[0] = h.result_shape[0] = multiply ? INT64_MAX : INT64_MAX / 16;
    h.Reject(state.get(), multiply ? "batch storage size overflows" : "addressable storage");
  }
  Harness h;
  h.Accept(state.get());  // Rejections never poison the prepared plan.
}

void InvalidStaticPlans() {
  const std::vector<std::vector<int64_t>> invalid{
      {}, {2, 8, 8, 0}, {1, -1, 8, 0}, {1, 8, 8, 1},
      {1, 8, 8, 1, 1, 0, 8, 1, 1, 1},
      {1, 8, 8, 1, 1, 8, 0, 1, 1, 1},
      {1, 8, 8, 1, 1, 0, 0, 3, 1, 0},
      {1, 8, 8, 1, 1, 0, 0, 3, INT64_MAX, 1},
      // A valid first record must not permit an invalid later record.
      {1, 8, 8, 2, 1, 0, 0, 2, 1, 1, 1, 2, 8, 2, 1, 1}};
  const auto created = Tensor0StrideCudaCopyPreparedCreatedCount();
  const auto destroyed = Tensor0StrideCudaCopyPreparedDestroyedCount();
  for (const auto& words : invalid) {
    auto state = native::InstantiateCopy({words.data(), words.size()});
    Require(!state.has_value(), "invalid static descriptor created a Copy state");
    Require(Tensor0StrideCudaCopyPreparedCreatedCount() == created &&
            Tensor0StrideCudaCopyPreparedDestroyedCount() == destroyed,
            "invalid static descriptor constructed then discarded a Copy state");
  }
}

__global__ void Produce(float* source) {
  const auto start = clock64();
  while (clock64() - start < 1000000) {}
  source[threadIdx.x] = float(20 + threadIdx.x);
}
__global__ void Consume(const float* result, float* observed) {
  observed[threadIdx.x] = result[threadIdx.x] + 1;
}
struct AddressPair { int64_t source, destination; };
__global__ void ProbeAddressPair(const int64_t* record, uint64_t logical,
                                 AddressPair* output) {
  int64_t source = record[2], destination = record[3];
  tensor0::stride::cuda::owner_fiber::Address(
      record, logical, false, source, destination);
  *output = {source, destination};
}

void AddressPairContract() {
  struct Case {
    std::vector<int64_t> record;
    uint64_t logical;
    AddressPair expected;
  };
  std::vector<int64_t> high_rank{9, 20, 100};
  high_rank.insert(high_rank.end(), 9, 2);
  for (int axis = 0; axis < 9; ++axis) high_rank.push_back(int64_t{1} << axis);
  for (int axis = 0; axis < 9; ++axis) high_rank.push_back(int64_t{1} << (8 - axis));
  const std::vector<Case> cases{
      {{0, 7, 9}, 0, {7, 9}},
      {{2, 10, 2, 1, 3, INT64_MAX, 2, INT64_MIN, 5}, 2, {14, 12}},
      {{2, 9, 17, 2, 3, -1, 4, 5, -2}, 4, {12, 20}},
      {{2, 3, 4, 3, 2, 0, 1, 0, 2}, 5, {4, 6}},
      {{2, 11, 12, 2, 3, 3, 1, 1, 2}, 4, {15, 15}},
      {high_rank, 3, {404, 103}},
  };
  int64_t* device_record = nullptr;
  AddressPair* device_result = nullptr;
  Cuda(cudaMalloc(reinterpret_cast<void**>(&device_record),
                  (high_rank.size() + 1) * sizeof(int64_t)));
  Cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(AddressPair)));
  for (const auto& test : cases) {
    std::vector<int64_t> words{1, 1024, 1024, 1};
    words.insert(words.end(), test.record.begin(), test.record.end());
    const auto decoded = tensor0::stride::descriptor::DecodeAddressLayout(
        words.data(), words.size());
    Require(tensor0::stride::layout::ElementCount(decoded.records[0]) > test.logical,
            "address probe logical index exceeds validated record");
    const auto packed = tensor0::stride::layout::PackOwnerFiber(decoded, 0);
    const auto* encoded = packed.words.data() + packed.schedules[0].cursor;
    const auto count = packed.words.size() - packed.schedules[0].cursor;
    Cuda(cudaMemcpy(device_record, encoded, count * sizeof(int64_t),
                    cudaMemcpyHostToDevice));
    ProbeAddressPair<<<1, 1>>>(device_record, test.logical, device_result);
    Cuda(cudaGetLastError());
    AddressPair actual{};
    Cuda(cudaMemcpy(&actual, device_result, sizeof(actual), cudaMemcpyDeviceToHost));
    Require(actual.source == test.expected.source &&
            actual.destination == test.expected.destination,
            "shared owner/fiber map address decoder mismatch");
  }
  // Empty records have zero logical elements, so neither map kernel calls the decoder.
  Cuda(cudaFree(device_result));
  Cuda(cudaFree(device_record));
}

void StreamAndStateLifetime() {
  Harness h;
  auto state = Prepare(kWords);
  auto* observed = reinterpret_cast<float*>(h.arena + 768);
  float* host = nullptr;
  Cuda(cudaMallocHost(reinterpret_cast<void**>(&host), 8 * sizeof(float)));
  Produce<<<1, 8, 0, h.stream>>>(static_cast<float*>(h.source.data));
  Cuda(cudaGetLastError());
  const auto error = h.Invoke(state.get());
  Require(!error.failure(), error.message());
  const auto destroyed = Tensor0StrideCudaCopyPreparedDestroyedCount();
  state.reset();  // No host state or host words are needed by pending kernels.
  Require(Tensor0StrideCudaCopyPreparedDestroyedCount() == destroyed + 1,
          "Copy state was not released before stream synchronization");
  Consume<<<1, 8, 0, h.stream>>>(static_cast<const float*>(h.result.data), observed);
  Cuda(cudaGetLastError());
  Cuda(cudaMemcpyAsync(host, observed, 8 * sizeof(float), cudaMemcpyDeviceToHost, h.stream));
  Cuda(cudaStreamSynchronize(h.stream));  // Only synchronization in this chain.
  for (int i = 0; i < 8; ++i)
    Require(host[i] == ((i == 2 || i == 4 || i == 6) ? 20 + i : 1),
            "producer/Copy/destroy/consumer ordering or device metadata lifetime failed");
  std::vector<int64_t> actual(h.descriptor_count);
  Cuda(cudaMemcpy(actual.data(), h.descriptor.data, actual.size() * 8, cudaMemcpyDeviceToHost));
  Require(actual == layout::PackOwnerFiber(
      descriptor::DecodeLayout(kWords.data(), kWords.size()), 0).words,
      "state destruction changed caller-owned device metadata");
  Cuda(cudaFreeHost(host));
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
    const auto created = Tensor0StrideCudaCopyPreparedCreatedCount();
    const auto destroyed = Tensor0StrideCudaCopyPreparedDestroyedCount();
    InvalidStaticPlans();
    AddressPairContract();
    ReuseAndHostLifetime();
    Rejections();
    StreamAndStateLifetime();
    Require(Tensor0StrideCudaCopyPreparedCreatedCount() - created ==
            Tensor0StrideCudaCopyPreparedDestroyedCount() - destroyed,
            "Copy prepared state leaked");
    std::cout << "CUDA Copy contract passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
