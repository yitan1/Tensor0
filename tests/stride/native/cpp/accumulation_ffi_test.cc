// Exercise the typed Accumulation handler and its shared cache on all worker paths.
#define TENSOR0_STRIDE_TEST_NO_HANDLERS
#include "ffi/reduction_impl.h"
#undef TENSOR0_STRIDE_TEST_NO_HANDLERS
#include "thread_pool_test_support.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace native = tensor0::stride;
extern "C" void Tensor0StrideSetWorkerLimit(uint64_t);

struct AccumulationFixture {
  const int64_t size;
  std::vector<int64_t> words;
  std::unique_ptr<native::PreparedState> state;
  testing::ThreadPool pool{4};
  std::array<int64_t, 2> source_dims, result_dims;
  int64_t factor_count = 1;
  float factor = 2;
  std::vector<float> factors, source, output;
  XLA_FFI_Buffer input, result, coefficient;
  XLA_FFI_ArgType kind = XLA_FFI_ArgType_BUFFER;
  void* pointer;
  XLA_FFI_Args args;
  int64_t record = 1;

  AccumulationFixture(int64_t n, int64_t batches) : size(n),
      words{1, n, n, 2, 1, 0, 0, n, 1, 1, 1, 0, 0, n, 1, 1},
      source_dims{batches, n}, result_dims{batches, n},
      factors(std::max<int64_t>(1, batches), 2),
      source(batches * n, 3), output(batches * n, 99),
      input{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
            source.data(), 2, source_dims.data()},
      result{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
             output.data(), 2, result_dims.data()},
      coefficient{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
                  &factor, 0, nullptr}, pointer(&coefficient),
      args{XLA_FFI_Args_STRUCT_SIZE, nullptr, 1, &kind, &pointer} {
    auto prepared = native::InstantiateAccumulation({words.data(), words.size()}, {&record, 1});
    assert(prepared.has_value());
    state = std::move(*prepared);
  }

  ffi::Future invoke(const native::PreparedState* prepared) {
    return native::Accumulation<ffi::F32>({words.data(), words.size()}, {&record, 1},
        prepared, ffi::AnyBuffer(&input), ffi::RemainingArgs(&args, 0),
        ffi::BufferR2<ffi::F32>(&result), pool.get());
  }
  ffi::Future invoke() { return invoke(state.get()); }
  void check(float value) const {
    for (float actual : output) assert(actual == value);
    for (float actual : source) assert(actual == 3);
  }
};

using Programs = std::vector<native::layout::GeneratedRecordProgram>;

void CheckProgramsUnchanged(const Programs& actual, const Programs& expected) {
  assert(actual.size() == expected.size());
  for (size_t index = 0; index < actual.size(); ++index) {
    const auto& a = actual[index];
    const auto& b = expected[index];
    assert(a.blocks == b.blocks && a.split_costs == b.split_costs);
    assert(a.record.semantic_index == b.record.semantic_index);
    assert(a.record.source_offset == b.record.source_offset);
    assert(a.record.destination_offset == b.record.destination_offset);
    assert(a.record.shape == b.record.shape);
    assert(a.record.source_strides == b.record.source_strides);
    assert(a.record.destination_strides == b.record.destination_strides);
  }
}

void CheckDisjointBranches() {
  constexpr int64_t width = 65536, size = 2 * width + 3;
  AccumulationFixture fixture(size, 1);
  fixture.words = {1, size, size, 2,
      1, 0, 1, 0, 1, 2, // Empty record preserves the next semantic index.
      1, 0, 1, width, 1, 2};
  assert(!native::InstantiateAccumulation(
      {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1}, 2).has_value());
  auto prepared = native::InstantiateAccumulation(
      {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1}, 1);
  assert(prepared.has_value());
  fixture.state = std::move(*prepared);
  Tensor0StrideSetWorkerLimit(4);
  fixture.factor = -1.25f;
  bool ready = false;
  testing::ObserveCompletion(native::Accumulation<ffi::F32>(
      {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1}, fixture.state.get(),
      ffi::AnyBuffer(&fixture.input), ffi::RemainingArgs(&fixture.args, 0),
      ffi::BufferR2<ffi::F32>(&fixture.result), fixture.pool.get(), 1), ready);
  assert(!ready && fixture.pool.tasks.size() == 2);
  fixture.pool.run_parallel();
  assert(ready);
  for (int64_t index = 0; index < size; ++index) {
    assert(fixture.output[index] == (index % 2 && index < 2 * width ? -3.75f : 0));
  }
  fixture.factor = 0;
  std::fill(fixture.source.begin(), fixture.source.end(), std::numeric_limits<float>::quiet_NaN());
  fixture.source[0] = std::numeric_limits<float>::infinity();
  ready = false;
  testing::ObserveCompletion(native::Accumulation<ffi::F32>(
      {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1}, fixture.state.get(),
      ffi::AnyBuffer(&fixture.input), ffi::RemainingArgs(&fixture.args, 0),
      ffi::BufferR2<ffi::F32>(&fixture.result), fixture.pool.get(), 1), ready);
  fixture.pool.run_parallel();
  assert(ready && fixture.output == std::vector<float>(size, 0));
  // No source access is permitted when every coefficient contribution is skipped.
  const auto programs = fixture.state->AccumulationPrograms(4, 4);
  assert(programs->size() == 1 && programs->front().record.semantic_index == 1);
  ready = false;
  testing::ObserveCompletion(native::ExecuteDisjointReduction<
      native::scalar::F32, native::scalar::F32>(fixture.pool.get(), programs,
      fixture.state->disjoint_work, nullptr, fixture.output.data(), size, size, 1,
      [](std::size_t, uint64_t, auto) {}), ready);
  assert(!ready && fixture.pool.tasks.size() == 2);
  fixture.pool.run_parallel();
  assert(ready && fixture.output == std::vector<float>(size, 0));
  AccumulationFixture batched(16384, 4);
  batched.words = {1, 16384, 16384, 2,
      1, 0, 0, 0, 1, 1, 1, 0, 0, 16384, 1, 1};
  auto batch_state = native::InstantiateAccumulation(
      {batched.words.data(), batched.words.size()}, {&batched.record, 1}, 1);
  assert(batch_state.has_value());
  batched.state = std::move(*batch_state);
  batched.factor_count = 4;
  batched.coefficient.rank = 1;
  batched.coefficient.dims = &batched.factor_count;
  batched.factors = {0, 1, -1.25f, 2};
  batched.coefficient.data = batched.factors.data();
  ready = false;
  testing::ObserveCompletion(native::Accumulation<ffi::F32>(
      {batched.words.data(), batched.words.size()}, {&batched.record, 1}, batched.state.get(),
      ffi::AnyBuffer(&batched.input), ffi::RemainingArgs(&batched.args, 0),
      ffi::BufferR2<ffi::F32>(&batched.result), batched.pool.get(), 1), ready);
  assert(!ready && batched.pool.tasks.size() > 1 && batched.pool.tasks.size() <= 4);
  batched.pool.run_parallel();
  assert(ready);
  for (int batch = 0; batch < 4; ++batch) {
    for (int index = 0; index < 16384; ++index) {
      assert(batched.output[batch * 16384 + index] == 3 * batched.factors[batch]);
    }
  }
}

void CheckDisjointOutputs() {
  constexpr int64_t width = 16384, count = 8, size = width * count + 3;
  AccumulationFixture fixture(size, 1);
  fixture.words = {1, size, size, count};
  for (int64_t index = 0; index < count; ++index) {
    fixture.words.insert(fixture.words.end(), {1, 0, index + 1, width, 1, count});
  }
  auto prepared = native::InstantiateAccumulation(
      {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1}, 1);
  assert(prepared.has_value());
  fixture.state = std::move(*prepared);
  assert(fixture.state->outputs_disjoint);
  const auto programs = fixture.state->AccumulationPrograms(4, 4);
  const auto snapshot = *programs;
  // The generic interval proof rejects these interleaved, independent maps.
  assert(native::layout::BuildReductionOutputTasks(*programs, 4).empty());
  for (uint64_t limit : {UINT64_C(1), UINT64_C(4), UINT64_C(1)}) {
    Tensor0StrideSetWorkerLimit(limit);
    for (float factor : {0.f, -1.25f, 2.f}) {
      fixture.factor = factor;
      std::fill(fixture.output.begin(), fixture.output.end(), 99);
      bool ready = false;
      testing::ObserveCompletion(native::Accumulation<ffi::F32>(
          {fixture.words.data(), fixture.words.size()}, {&fixture.record, 1},
          fixture.state.get(), ffi::AnyBuffer(&fixture.input),
          ffi::RemainingArgs(&fixture.args, 0), ffi::BufferR2<ffi::F32>(&fixture.result),
          fixture.pool.get(), 1), ready);
      if (limit == 4) {
        assert(!ready && fixture.pool.tasks.size() == 4);
        fixture.pool.run_parallel();
      }
      assert(ready && fixture.pool.tasks.empty());
      for (int64_t index = 0; index < size; ++index) {
        const float expected = index == 0 || index > width * count ? 0 :
            3 * ((index - 1) % count == fixture.record ? factor : 1);
        assert(fixture.output[index] == expected);
      }
      CheckProgramsUnchanged(*programs, snapshot);
    }
  }
  Tensor0StrideSetWorkerLimit(4);
  std::weak_ptr<const Programs> lifetime;
  {
    // The queued handler must own its immutable programs, not PreparedState.
    auto owner = fixture.state->AccumulationPrograms(8, 4);
    lifetime = owner;
    std::vector<double> source(size, 1.00000003);
    std::vector<float> expected(size);
    auto bind = [&](std::size_t index, uint64_t batch, auto apply) {
      assert(batch == 0);
      apply(native::expression::Scale<native::scalar::F64>{index == 1 ? 1.25 : 1});
    };
    native::ExecuteReductionBatch<native::scalar::F64, native::scalar::F32>(
        *owner, source.data(), expected.data(), size,
        [&](std::size_t index, auto apply) { bind(index, 0, apply); });
    bool ready = false;
    testing::ObserveCompletion(native::ExecuteDisjointReduction<
        native::scalar::F64, native::scalar::F32>(fixture.pool.get(), owner,
        fixture.state->disjoint_work, source.data(), fixture.output.data(),
        size, size, 1, bind), ready);
    assert(!ready && fixture.pool.tasks.size() == 4);
    owner.reset();
    fixture.state.reset();
    assert(!lifetime.expired());
    fixture.pool.run_parallel();
    assert(ready && lifetime.expired() && fixture.output == expected);
  }
}

int main() {
  CheckDisjointOutputs();
  CheckDisjointBranches();
  Tensor0StrideSetWorkerLimit(1);
  {
    AccumulationFixture fixture(3, 0);
    assert(testing::CompletedError(fixture.invoke()).success());
    assert(fixture.pool.tasks.empty() && fixture.output.empty());
    fixture.factor_count = 2;
    fixture.coefficient.rank = 1;
    fixture.coefficient.dims = &fixture.factor_count;
    assert(testing::CompletedError(fixture.invoke()).message().find("batch count") != std::string::npos);
  }
  {
    AccumulationFixture fixture(0, 3);
    assert(testing::CompletedError(fixture.invoke()).success());
    assert(fixture.pool.tasks.empty() && fixture.output.empty());
  }
  {
    AccumulationFixture fixture(3, 3);
    for (float factor : {2.f, 0.f, -1.f, 4.f}) {
      fixture.factor = factor;
      std::fill(fixture.output.begin(), fixture.output.end(), 99);
      assert(testing::CompletedError(fixture.invoke()).success());
      assert(fixture.pool.tasks.empty());
      fixture.check(3 * (1 + factor));
    }
    fixture.coefficient.rank = 1;
    fixture.factor_count = 3;
    fixture.coefficient.dims = &fixture.factor_count;
    fixture.coefficient.data = fixture.factors.data();
    fixture.factors = {0, 2, -1};
    assert(testing::CompletedError(fixture.invoke()).success());
    for (int batch = 0; batch < 3; ++batch) {
      for (int index = 0; index < 3; ++index) {
        assert(fixture.output[batch * 3 + index] == 3 * (1 + fixture.factors[batch]));
      }
    }
    const auto old = fixture.output;
    const auto reject = [&](const char* message) {
      const auto error = testing::CompletedError(fixture.invoke());
      assert(error.failure() && error.message().find(message) != std::string::npos);
      assert(fixture.output == old && fixture.pool.tasks.empty());
    };
    fixture.source_dims[1] = 2; reject("dimensions"); fixture.source_dims[1] = 3;
    fixture.result_dims[0] = 2; reject("dimensions"); fixture.result_dims[0] = 3;
    fixture.source_dims[0] = -1; reject("nonnegative"); fixture.source_dims[0] = 3;
    fixture.input.rank = 1; reject("rank-two"); fixture.input.rank = 2;
    fixture.factor_count = 2; reject("batch count"); fixture.factor_count = 3;
    fixture.result.data = fixture.source.data(); reject("overlap"); fixture.result.data = fixture.output.data();
    fixture.coefficient.data = fixture.output.data(); reject("overlap");
    fixture.coefficient.data = fixture.factors.data();
    fixture.record = 2; reject("indices"); fixture.record = 1;
    assert(testing::CompletedError(fixture.invoke()).success());
  }
  {
    AccumulationFixture fixture(3, 1);
    std::atomic<int> start{0};
    std::array<bool, 12> good{};
    std::vector<std::thread> threads;
    for (int index = 0; index < 12; ++index) threads.emplace_back([&, index] {
      std::array<float, 3> source{float(index + 1), 2, 3}, output{99, 99, 99};
      std::array<int64_t, 2> dims{1, 3};
      XLA_FFI_Buffer input{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
                           source.data(), 2, dims.data()};
      XLA_FFI_Buffer result{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
                            output.data(), 2, dims.data()};
      float factor = index % 2 ? 0.f : 2.f;
      XLA_FFI_Buffer coefficient{XLA_FFI_Buffer_STRUCT_SIZE, nullptr, XLA_FFI_DataType_F32,
                                  &factor, 0, nullptr};
      XLA_FFI_ArgType kind = XLA_FFI_ArgType_BUFFER;
      void* pointer = &coefficient;
      XLA_FFI_Args args{XLA_FFI_Args_STRUCT_SIZE, nullptr, 1, &kind, &pointer};
      start.fetch_add(1);
      while (start.load() != 12) std::this_thread::yield();
      auto future = native::Accumulation<ffi::F32>({fixture.words.data(), fixture.words.size()},
          {&fixture.record, 1}, fixture.state.get(), ffi::AnyBuffer(&input),
          ffi::RemainingArgs(&args, 0), ffi::BufferR2<ffi::F32>(&result), fixture.pool.get());
      good[index] = testing::CompletedError(std::move(future)).success() &&
          output == std::array<float, 3>{source[0] * (1 + factor), 2 * (1 + factor), 3 * (1 + factor)};
    });
    for (auto& thread : threads) thread.join();
    for (bool ok : good) assert(ok);
    assert(fixture.pool.tasks.empty());
  }
  {
    AccumulationFixture fixture(65536, 1);
    const auto programs = fixture.state->AccumulationPrograms(4, 4);
    const auto snapshot = *programs;
    for (uint64_t limit : {UINT64_C(1), UINT64_C(4), UINT64_C(1)}) {
      Tensor0StrideSetWorkerLimit(limit);
      assert(native::AvailableWorkerCount(fixture.pool.get()) == limit);
      std::fill(fixture.output.begin(), fixture.output.end(), 99);
      auto future = fixture.invoke();
      bool ready = false;
      future.OnReady([&](const std::optional<ffi::Error>& error) { assert(!error); ready = true; });
      if (limit == 4) {
        assert(!ready && fixture.pool.tasks.size() > 1);
        fixture.pool.run_parallel();
      } else {
        assert(ready && fixture.pool.tasks.empty());
      }
      assert(ready);
      fixture.check(9);
      assert(programs == fixture.state->AccumulationPrograms(4, 4));
      assert(programs.use_count() == 2);
      CheckProgramsUnchanged(*programs, snapshot);
    }
  }
  {
    // Concurrent first calls must share the cache, not buffers or task pools.
    Tensor0StrideSetWorkerLimit(4);
    AccumulationFixture shared(16384, 4);
    constexpr int callers = 4;
    std::atomic<int> queued{0};
    std::atomic<bool> run{false};
    std::array<bool, callers> good{};
    std::vector<std::thread> threads;
    for (int index = 0; index < callers; ++index) threads.emplace_back([&, index] {
      AccumulationFixture local(shared.size, 4);
      local.state.reset();
      local.factor = float(index - 1);
      auto future = local.invoke(shared.state.get());
      bool ready = false;
      testing::ObserveCompletion(std::move(future), ready);
      assert(!ready && local.pool.tasks.size() > 1);
      queued.fetch_add(1);
      while (!run.load()) std::this_thread::yield();
      local.pool.run_parallel();
      local.check(3 * (1 + local.factor));
      good[index] = ready;
    });
    while (queued.load() != callers) std::this_thread::yield();
    const auto programs = shared.state->AccumulationPrograms(4, 4);
    // One owner in PreparedState, one here, and at least one per queued call.
    assert(programs.use_count() >= callers + 2);
    run = true;
    for (auto& thread : threads) thread.join();
    for (bool ok : good) assert(ok);
    assert(programs.use_count() == 2);
    assert(programs == shared.state->AccumulationPrograms(4, 4));
  }
  {
    // Queued batch tasks retain the cache after PreparedState is destroyed.
    AccumulationFixture fixture(16384, 4);
    auto future = fixture.invoke();
    bool ready = false;
    testing::ObserveCompletion(std::move(future), ready);
    assert(!ready && fixture.pool.tasks.size() > 1);
    std::weak_ptr<const Programs> lifetime = fixture.state->AccumulationPrograms(4, 4);
    fixture.state.reset();
    assert(!lifetime.expired());
    fixture.pool.run_parallel();
    assert(ready && lifetime.expired());
    fixture.check(9);
  }
  {
    // Partials rebase a nonzero destination only in private split programs.
    AccumulationFixture fixture(65536, 1);
    native::descriptor::DecodedLayout decoded;
    decoded.source_size = decoded.output_size = fixture.size;
    decoded.records = {
        {0, 0, 2, {uint64_t(fixture.size)}, {1}, {0}},
        {1, 0, 2, {uint64_t(fixture.size)}, {1}, {0}}};
    fixture.state = std::make_unique<native::PreparedState>(std::move(decoded));
    const auto programs = fixture.state->AccumulationPrograms(4, 4);
    const auto snapshot = *programs;
    {
      const auto plan = native::PrepareReductionPlan(fixture.pool.get(), fixture.state->records,
          native::ReductionKind::Reduce, true, fixture.size, 0, fixture.size, 1, 4, 0, 4, programs);
      assert(plan.mode == native::ReductionMode::Partials && plan.output_index == 2);
      assert(!plan.shared_programs);
      CheckProgramsUnchanged(*programs, snapshot);
    }
    for (uint64_t limit : {UINT64_C(4), UINT64_C(1), UINT64_C(4)}) {
      Tensor0StrideSetWorkerLimit(limit);
      fixture.factor = limit == 1 ? 0.f : 2.f;
      std::fill(fixture.output.begin(), fixture.output.end(), 99);
      bool ready = false;
      testing::ObserveCompletion(fixture.invoke(), ready);
      if (limit == 4) {
        assert(!ready && fixture.pool.tasks.size() > 1);
        fixture.pool.run_parallel();
      }
      assert(ready);
      for (size_t index = 0; index < fixture.output.size(); ++index) {
        assert(fixture.output[index] == (index == 2 ? 3 * fixture.size * (1 + fixture.factor) : 0));
      }
      assert(programs == fixture.state->AccumulationPrograms(4, 4));
      CheckProgramsUnchanged(*programs, snapshot);
    }
  }
  {
    // Internal preparation failure must become a failed Future on either path.
    AccumulationFixture fixture(1, 1);
    native::descriptor::DecodedLayout bad;
    bad.source_size = bad.output_size = 1;
    bad.records.push_back({0, 0, 0, {UINT64_MAX, 2}, {0, 0}, {0, 0}});
    native::PreparedState malformed(std::move(bad));
    fixture.args.size = 0;
    for (uint64_t limit : {UINT64_C(1), UINT64_C(4)}) {
      Tensor0StrideSetWorkerLimit(limit);
      auto future = native::Accumulation<ffi::F32>({fixture.words.data(), fixture.words.size()},
          {}, &malformed, ffi::AnyBuffer(&fixture.input), ffi::RemainingArgs(&fixture.args, 0),
          ffi::BufferR2<ffi::F32>(&fixture.result), fixture.pool.get());
      const auto error = testing::CompletedError(std::move(future));
      assert(error.failure() && error.message().find("layout element count overflows") != std::string::npos);
      assert(fixture.pool.tasks.empty() && fixture.output[0] == 99);
    }
  }
  Tensor0StrideSetWorkerLimit(UINT64_MAX);
}
