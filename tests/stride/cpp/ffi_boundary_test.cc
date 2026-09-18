#include "ffi_under_test.inc"
#include "ffi_test_support.h"

#include <cassert>

namespace native = tensor0::stride;

void CheckBatchScheduling() {
  for (int64_t threads : {0, 1, 4}) {
    for (uint64_t batches : {UINT64_C(0), UINT64_C(1), UINT64_C(7)}) {
      for (uint64_t elements : {UINT64_C(0), UINT64_C(3), UINT64_C(65536)}) {
        testing::ThreadPool pool(threads);
        std::vector<int> visits(batches, 0);
        auto owner = std::make_shared<int>(42);
        std::weak_ptr<int> lifetime = owner;
        auto future = native::ExecuteBatchTasks(pool.get(), batches, elements,
            [owner, &visits](uint64_t begin, uint64_t count) {
              assert(*owner == 42);
              for (uint64_t batch = begin; batch < begin + count; ++batch) ++visits[batch];
            });
        owner.reset();
        bool ready = false;
        future.OnReady([&](const std::optional<ffi::Error>& error) {
          assert(!error);
          ready = true;
        });
        const bool parallel = threads == 4 && batches == 7 && elements == 65536;
        assert(ready != parallel);
        assert(pool.tasks.size() == (parallel ? 4 : 0));
        if (parallel) {
          assert(!lifetime.expired());
          pool.run_one();
          assert(!ready);
          pool.run_parallel();
        }
        assert(ready && lifetime.expired());
        for (int count : visits) assert(count == (elements == 0 ? 0 : 1));
      }
    }
  }
  for (bool unknown : {false, true}) {
    testing::ThreadPool pool(4);
    auto future = native::ExecuteBatchTasks(pool.get(), 8, 65536,
        [unknown](uint64_t begin, uint64_t) {
          if (begin == 6) {
            if (unknown) throw 1;
            throw std::runtime_error("injected worker failure");
          }
        });
    bool ready = false;
    future.OnReady([&](const std::optional<ffi::Error>& error) {
      assert(error && error->failure());
      ready = true;
    });
    pool.run_one();
    assert(!ready);
    pool.run_parallel();
    assert(ready);
  }
}

void CheckParallelUpdate() {
  constexpr int64_t batches = 9, size = 16384, selected = 4096;
  const std::vector<int64_t> words{1, size, size, 1, 1, 1, 1, selected, 2, 2};
  auto prepared = native::InstantiatePrepared(words);
  assert(prepared.has_value());
  testing::ThreadPool pool(4);
  std::vector<float> source(batches * size, 3), base(batches * size, 5);
  std::vector<float> result(batches * size, 99), second(batches * size, 99);
  std::array<float, batches> alpha{}, beta{};
  for (int64_t batch = 0; batch < batches; ++batch) {
    alpha[batch] = batch % 3;
    beta[batch] = (batch + 1) % 3;
  }
  const auto invoke = [&](const float* input, float* output, int64_t alpha_count, int64_t beta_count) {
    std::array<int64_t, 2> shape{batches, size};
    XLA_FFI_Buffer source_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
        const_cast<float*>(input), 2, shape.data()};
    XLA_FFI_Buffer base_buffer = source_buffer;
    base_buffer.data = base.data();
    XLA_FFI_Buffer output_buffer = base_buffer;
    output_buffer.data = output;
    XLA_FFI_Buffer alpha_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
        alpha.data(), 1, &alpha_count};
    XLA_FFI_Buffer beta_buffer = alpha_buffer;
    beta_buffer.data = beta.data();
    beta_buffer.dims = &beta_count;
    return native::Update<ffi::F32>({}, prepared->get(), ffi::AnyBuffer(&source_buffer),
        ffi::BufferR2<ffi::F32>(&base_buffer), ffi::AnyBuffer(&alpha_buffer),
        ffi::AnyBuffer(&beta_buffer), ffi::BufferR2<ffi::F32>(&output_buffer), pool.get());
  };
  for (int mode = 0; mode < 3; ++mode) {
    for (int64_t alpha_count : {INT64_C(1), batches}) {
      for (int64_t beta_count : {INT64_C(1), batches}) {
        std::fill(base.begin(), base.end(), 5);
        std::fill(result.begin(), result.end(), 99);
        const auto* input = mode == 2 ? base.data() : source.data();
        auto* output = mode == 0 ? result.data() : base.data();
        auto future = invoke(input, output, alpha_count, beta_count);
        assert(pool.tasks.size() == 4);
        bool ready = false;
        future.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); ready = true; });
        assert(!ready);
        for (float value : result) assert(value == 99);
        for (float value : base) assert(value == 5);
        bool second_ready = mode != 0;
        if (mode == 0) {
          auto another = invoke(input, second.data(), alpha_count, beta_count);
          another.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); second_ready = true; });
        }
        pool.run_parallel();
        assert(ready && second_ready);
        for (int64_t batch = 0; batch < batches; ++batch) {
          for (int64_t index = 0; index < size; ++index) {
            const float expected = index % 2 == 1 && index < 2 * selected
                ? alpha[alpha_count == 1 ? 0 : batch] * (mode == 2 ? 5 : 3) +
                    beta[beta_count == 1 ? 0 : batch] * 5 : 5;
            assert(output[batch * size + index] == expected);
            if (mode == 0) assert(second[batch * size + index] == expected);
          }
        }
      }
    }
  }
  const auto original = result;
  auto invalid = invoke(source.data(), result.data(), 3, batches);
  bool rejected = false;
  invalid.OnReady([&](const std::optional<ffi::Error>& error) {
    assert(error && error->message().find("coefficients") != std::string::npos);
    rejected = true;
  });
  assert(rejected && pool.tasks.empty() && result == original);
}

std::vector<uint8_t> Bytes(std::initializer_list<uint64_t> words) {
  std::vector<uint8_t> bytes;
  for (const auto word : words) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      bytes.push_back(static_cast<uint8_t>(word >> shift));
    }
  }
  return bytes;
}

void CheckReductionBoundary() {
  const auto bytes = Bytes({1, 3, 2, 1, 1, 0, 1, 3, 1, 1, 1, 1});
  std::array<int64_t, 2> indices{0, 0};
  const ffi::Span<const uint8_t> layout(bytes.data(), bytes.size());
  auto state = native::InstantiateReduction(layout, {});
  assert(state.has_value());
  static_assert(std::is_same_v<decltype(*state), std::unique_ptr<native::PreparedState>&>);
  std::array<double, 3> source{1.5, 2.5, 3.5};
  std::array<uint8_t, 2> output{99, 99};
  float coefficient = 2;
  std::array<int64_t, 2> source_dims{1, 3};
  std::array<int64_t, 2> result_dims{1, 2};
  XLA_FFI_Buffer source_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_F64, source.data(), 2, source_dims.data()};
  XLA_FFI_Buffer result_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_U8, output.data(), 2, result_dims.data()};
  XLA_FFI_Buffer coefficient_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_F32, &coefficient, 0, nullptr};
  std::array<XLA_FFI_ArgType, 2> kinds{XLA_FFI_ArgType_BUFFER, XLA_FFI_ArgType_BUFFER};
  std::array<void*, 2> pointers{&coefficient_buffer, &coefficient_buffer};
  XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, 1, kinds.data(), pointers.data()};
  std::size_t index_count = 1;
  testing::ThreadPool pool;
  const auto invoke = [&] {
    return testing::CompletedError(native::Reduction<ffi::U8>(layout, {indices.data(), index_count}, state->get(),
        ffi::AnyBuffer(&source_buffer), ffi::RemainingArgs(&args, 0),
        ffi::BufferR2<ffi::U8>(&result_buffer), pool.get()));
  };
  assert(invoke().success());
  assert((output == std::array<uint8_t, 2>{0, 15}));
  coefficient = 4;
  assert(invoke().success());
  assert((output == std::array<uint8_t, 2>{0, 30}));
  const auto unchanged = output;
  const auto before_source = source;
  result_buffer.data = reinterpret_cast<uint8_t*>(source.data()) + 4;
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(source == before_source);
  result_buffer.data = reinterpret_cast<uint8_t*>(&coefficient) + 1;
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(coefficient == 4);
  result_buffer.data = output.data();
  const auto reject = [&](const char* message) {
    assert(invoke().message().find(message) != std::string::npos);
    assert(output == unchanged);
  };
  args.size = 0;
  reject("count");
  args.size = 1;
  indices[0] = -1;
  reject("indices");
  indices[0] = 1;
  reject("indices");
  indices[0] = 0;
  args.size = 2;
  index_count = 2;
  reject("indices");
  args.size = 1;
  index_count = 1;
  coefficient_buffer.rank = 1;
  coefficient_buffer.dims = result_dims.data() + 1;
  reject("batch count");
  coefficient_buffer.rank = 2;
  coefficient_buffer.dims = result_dims.data();
  reject("batch count");
  coefficient_buffer.rank = 0;
  coefficient_buffer.dims = nullptr;
  coefficient_buffer.dtype = XLA_FFI_DataType_TOKEN;
  reject("unsupported scalar dtype");
  coefficient_buffer.dtype = XLA_FFI_DataType_F32;
  source_buffer.dtype = XLA_FFI_DataType_TOKEN;
  reject("unsupported scalar dtype");
  source_buffer.dtype = XLA_FFI_DataType_F64;
  source_buffer.rank = 1;
  reject("rank-two");
  source_buffer.rank = 2;
  source_dims[0] = -1;
  reject("nonnegative");
  source_dims[0] = 2;
  reject("dimensions");
  source_dims[0] = INT64_MAX;
  result_dims[0] = INT64_MAX;
  reject("overflows");
  source_dims[0] = 0;
  result_dims[0] = 0;
  assert(invoke().success());
  assert(output == unchanged);
  coefficient_buffer.rank = 1;
  coefficient_buffer.dims = source_dims.data();
  coefficient_buffer.data = nullptr;
  assert(invoke().success());
  assert(output == unchanged);
  coefficient_buffer.dtype = XLA_FFI_DataType_TOKEN;
  reject("unsupported scalar dtype");
  coefficient_buffer.dtype = XLA_FFI_DataType_F32;
  std::array<float, 3> factors{0, 1, 2};
  const auto original_factors = factors;
  source_dims[0] = 3;
  result_dims[0] = 3;
  coefficient_buffer.data = factors.data();
  result_buffer.data = reinterpret_cast<uint8_t*>(factors.data() + 2);
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(factors == original_factors);
}


void CheckDotBoundary() {
  testing::ThreadPool pool;
  const std::vector<int64_t> words{1, 3, 5, 1, 1, 0, 1, 3, 1, 1};
  const ffi::Span<const int64_t> layout(words.data(), words.size());
  auto state = native::InstantiateDot(layout, 0);
  assert(state.has_value());
  assert((*state)->source_size == 3 && (*state)->output_size == 5);
  std::array<float, 3> left{1, 2, 3};
  std::array<float, 5> right{0, 4, 5, 6, 0};
  std::array<float, 3> output{7, 7, 7};
  std::array<int64_t, 2> left_dims{1, 3}, right_dims{1, 5};
  std::array<int64_t, 1> result_dims{1};
  XLA_FFI_Buffer left_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_F32, left.data(), 2, left_dims.data()};
  XLA_FFI_Buffer right_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_F32, right.data(), 2, right_dims.data()};
  XLA_FFI_Buffer result_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr,
      XLA_FFI_DataType_F32, output.data() + 1, 1, result_dims.data()};
  const auto invoke = [&](int64_t conjugate = 0) {
    return testing::CompletedError(native::Dot<ffi::F32>(layout, conjugate, state->get(),
        ffi::AnyBuffer(&left_buffer), ffi::AnyBuffer(&right_buffer),
        ffi::BufferR1<ffi::F32>(&result_buffer), pool.get()));
  };
  assert(invoke().success());
  assert((output == std::array<float, 3>{7, 32, 7}));
  assert(invoke(1).success());
  assert((output == std::array<float, 3>{7, 32, 7}));
  const auto original_left = left;
  const auto original_right = right;
  result_buffer.data = left.data() + 2;
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(left == original_left);
  result_buffer.data = right.data() + 4;
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(right == original_right);
  result_buffer.data = output.data() + 1;
  const auto reject = [&](const char* message) {
    assert(invoke().message().find(message) != std::string::npos);
    assert((output == std::array<float, 3>{7, 32, 7}));
  };
  assert(invoke(2).message().find("conjugate_left") != std::string::npos);
  assert((output == std::array<float, 3>{7, 32, 7}));
  left_dims[0] = -1;
  reject("nonnegative");
  left_dims[0] = 1;
  right_dims[1] = 3;
  reject("dimensions");
  right_dims[1] = 5;
  right_dims[0] = 2;
  reject("dimensions");
  right_dims[0] = 1;
  left_buffer.rank = 1;
  reject("rank-two");
  left_buffer.rank = 2;
  right_buffer.dtype = XLA_FFI_DataType_F8E4M3FN;
  reject("dtype");
  right_buffer.dtype = XLA_FFI_DataType_F32;
  std::array<std::complex<double>, 3> complex_left{{{1, 2}, {2, 3}, {3, 4}}};
  const auto original_complex = complex_left;
  left_buffer.dtype = XLA_FFI_DataType_C128;
  left_buffer.data = complex_left.data();
  result_buffer.data = reinterpret_cast<char*>(complex_left.data()) + sizeof(complex_left) - sizeof(float);
  assert(invoke().message().find("overlap") != std::string::npos);
  assert(complex_left == original_complex);
  result_buffer.data = output.data() + 1;
  assert(invoke().success());
  assert((output == std::array<float, 3>{7, 32, 7}));
  left_buffer.dtype = XLA_FFI_DataType_F32;
  left_buffer.data = left.data();
  result_dims[0] = 2;
  reject("dimensions");
  result_dims[0] = 1;
  left_dims[0] = right_dims[0] = result_dims[0] = INT64_MAX;
  reject("overflows");
  left_dims[0] = right_dims[0] = result_dims[0] = 0;
  left_buffer.data = right_buffer.data = result_buffer.data = nullptr;
  assert(invoke().success());
  assert((output == std::array<float, 3>{7, 32, 7}));
  const std::vector<int64_t> huge{1, INT64_MAX, 0, 0};
  const ffi::Span<const int64_t> huge_layout(huge.data(), huge.size());
  auto large_state = native::InstantiateDot(huge_layout, 0);
  assert(large_state.has_value());
  left_dims = {1, INT64_MAX};
  right_dims = {1, 0};
  result_dims[0] = 1;
  result_buffer.data = output.data() + 1;
  const auto error = testing::CompletedError(native::Dot<ffi::F32>(huge_layout, 0, large_state->get(),
      ffi::AnyBuffer(&left_buffer), ffi::AnyBuffer(&right_buffer),
      ffi::BufferR1<ffi::F32>(&result_buffer), pool.get()));
  assert(error.message().find("addressable") != std::string::npos);
  assert((output == std::array<float, 3>{7, 32, 7}));
}


int main() {
  CheckReductionBoundary();
  CheckDotBoundary();
  CheckBatchScheduling();
  CheckParallelUpdate();
  const std::vector<int64_t> words{1, 8, 8, 1, 1, 1, 1, 3, 2, 2};
  auto prepared = native::InstantiatePrepared(words);
  assert(prepared.has_value());
  std::array<float, 16> source, base, result;
  source.fill(3);
  base.fill(5);
  result.fill(99);
  float alpha = 2, beta = 3;
  std::array<int64_t, 2> shape{2, 8}, output_shape{2, 8};
  int64_t count = 1;
  XLA_FFI_Buffer source_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
      source.data(), 2, shape.data()};
  XLA_FFI_Buffer base_buffer = source_buffer;
  base_buffer.data = base.data();
  base_buffer.dims = output_shape.data();
  XLA_FFI_Buffer output = base_buffer;
  output.data = result.data();
  XLA_FFI_Buffer alpha_buffer{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
      &alpha, 1, &count};
  XLA_FFI_Buffer beta_buffer = alpha_buffer;
  beta_buffer.data = &beta;
  const auto invoke = [&]() {
    return testing::Update<ffi::F32>(ffi::Span<const int64_t>{}, prepared->get(),
        ffi::AnyBuffer(&source_buffer), ffi::BufferR2<ffi::F32>(&base_buffer),
        ffi::AnyBuffer(&alpha_buffer), ffi::AnyBuffer(&beta_buffer),
        ffi::BufferR2<ffi::F32>(&output));
  };
  for (int mode = 0; mode < 3; ++mode) {
    source.fill(3);
    base.fill(5);
    result.fill(99);
    source_buffer.data = mode == 2 ? base.data() : source.data();
    output.data = mode == 0 ? result.data() : base.data();
    assert(invoke().success());
    const auto& actual = mode == 0 ? result : base;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto position = index % 8;
      const bool selected = position == 1 || position == 3 || position == 5;
      assert(actual[index] == (selected ? (mode == 2 ? 25 : 21) : 5));
    }
    for (float value : source) assert(value == 3);
  }
  source.fill(3);
  base.fill(5);
  result.fill(99);
  source_buffer.data = source.data();
  output.data = base.data();
  const auto reject = [&](const std::string& message) {
    const auto old_source = source, old_base = base, old_result = result;
    const auto error = invoke();
    assert(error.failure() && error.message().find(message) != std::string::npos);
    assert(source == old_source && base == old_base && result == old_result);
  };
  output.data = base.data() + 1;
  reject("overlap");
  output.data = base.data();
  source_buffer.data = base.data() + 1;
  reject("overlap");
  source_buffer.data = source.data();
  output.data = source.data();
  reject("alias");
  source_buffer.data = base.data();
  output.data = base.data();
  source_buffer.dtype = XLA_FFI_DataType_C64;
  reject("alias");
  source_buffer.dtype = XLA_FFI_DataType_F32;
  alpha_buffer.data = base.data() + 7;
  reject("overlap");
  alpha_buffer.data = &alpha;
  beta_buffer.data = base.data();
  reject("overlap");
  beta_buffer.data = &beta;
  count = 3;
  reject("coefficients");
  count = 1;
  source_buffer.dtype = XLA_FFI_DataType_F32;
  shape[0] = -1;
  reject("nonnegative");
  shape[0] = 1;
  reject("dimensions");
  shape[0] = output_shape[0] = INT64_MAX;
  reject("overflows");
  shape[0] = output_shape[0] = INT64_MAX / 8;
  reject("addressable");
  shape[0] = output_shape[0] = 2;
  const std::vector<int64_t> late_invalid{1, 8, 8, 2, 0, 1, 1, 1, 0, 3, 2, 1, 2};
  prepared = native::InstantiatePrepared(late_invalid);
  assert(prepared.has_value());
  reject("per-element addresses");
  const std::vector<int64_t> empty{1, 8, 8, 0};
  prepared = native::InstantiatePrepared(empty);
  assert(invoke().success());
  for (float value : base) assert(value == 5);
}
