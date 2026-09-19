#include <atomic>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "layout/blocking.inc"
#include "ffi/descriptor.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "execute/scheduling.inc"
#include "execute/reduction.inc"
#include "thread_pool_test_support.h"

std::vector<std::pair<int64_t, int64_t>> Addresses(const layout::Record& record) {
  std::vector<std::pair<int64_t, int64_t>> result;
  for (uint64_t element = 0; element < layout::ElementCount(record); ++element) {
    auto remaining = element;
    auto source = record.source_offset;
    auto destination = record.destination_offset;
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      const auto coordinate = static_cast<int64_t>(remaining % record.shape[axis]);
      remaining /= record.shape[axis];
      source += coordinate * record.source_strides[axis];
      destination += coordinate * record.destination_strides[axis];
    }
    result.emplace_back(source, destination);
  }
  return result;
}

void CheckKernel(const layout::Record& record) {
  const auto validated = layout::BuildLayout(record.shape,
      record.source_strides, record.source_offset, record.destination_strides,
      record.destination_offset, 64, 64, record.semantic_index);
  assert(validated.shape == record.shape && validated.source_strides == record.source_strides &&
         validated.destination_strides == record.destination_strides &&
         validated.source_offset == record.source_offset && validated.destination_offset == record.destination_offset);
  std::vector<int32_t> source(64);
  for (std::size_t index = 0; index < source.size(); ++index) source[index] = static_cast<int32_t>(index % 7) - 3;
  const auto original = source;
  std::vector<int32_t> expected(66, 7);
  for (const auto& [source_index, destination_index] : Addresses(record)) {
    expected[destination_index + 1] += source[source_index];
  }
  for (bool optimize : {false, true}) {
    auto planned = validated;
    if (optimize) layout::OptimizeRecordForExecution(&planned);
    auto before = Addresses(record);
    auto after = Addresses(planned);
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    assert(before == after);
    assert(planned.semantic_index == record.semantic_index);
    std::vector<int32_t> output(66, 7);
    kernels::ExecuteReductionRecord<scalar::S32>(
        planned, source.data(), output.data() + 1, expression::Identity<scalar::S32>{});
    assert(output == expected);
    assert(source == original);
  }
}

void CheckLayouts() {
  CheckKernel({7, 0, 1, {2, 2}, {1, 2}, {1, 1}});
  CheckKernel({7, 7, 2, {2, 2, 2}, {-1, -2, -4}, {1, -1, 0}});
  CheckKernel({7, 0, 1, {2, 2, 2}, {1, 2, 4}, {0, 1, 1}});
  CheckKernel({7, 64, 64, {0}, {INT64_MIN}, {INT64_MAX}});
  CheckKernel({7, 2, 2, {1}, {INT64_MIN}, {INT64_MAX}});
  CheckKernel({7, 2, 2, {}, {}, {}});
  CheckKernel({7, 2, 2, std::vector<uint64_t>(70, 1),
      std::vector<int64_t>(70, INT64_MIN), std::vector<int64_t>(70, INT64_MAX)});
  std::mt19937 random(71);
  for (std::size_t trial = 0; trial < 1000; ++trial) {
    layout::Record record{trial, 24, 24, {}, {}, {}};
    for (std::size_t axis = 0; axis < 3; ++axis) {
      record.shape.push_back(random() % 4);
      record.source_strides.push_back(static_cast<int64_t>(random() % 7) - 3);
      record.destination_strides.push_back(static_cast<int64_t>(random() % 7) - 3);
    }
    CheckKernel(record);
  }
}

void CheckExecution() {
  const std::vector<layout::Record> records{
      {7, 0, 1, {2, 2}, {1, 2}, {1, 1}},
      {8, 4, 5, {0}, {1}, {1}},
      {9, 3, 3, {2, 2}, {-1, -2}, {-1, -1}}};
  for (bool optimize : {false, true}) {
    auto planned = records;
    for (auto& record : planned) {
      record = layout::BuildLayout(record.shape, record.source_strides,
          record.source_offset, record.destination_strides, record.destination_offset,
          4, 5, record.semantic_index);
    }
    if (optimize) for (auto& record : planned) layout::OptimizeRecordForExecution(&record);
    for (uint64_t batches : {UINT64_C(0), UINT64_C(1), UINT64_C(3)}) {
      std::vector<float> source(batches * 4);
      for (std::size_t index = 0; index < source.size(); ++index) source[index] = index + 1;
      const auto original = source;
      std::vector<float> expected(batches * 5 + 2, 0);
      expected.front() = expected.back() = 7;
      for (const auto& record : records) {
        for (uint64_t batch = 0; batch < batches; ++batch) {
          for (const auto& [source_index, destination_index] : Addresses(record)) {
            expected[1 + batch * 5 + destination_index] +=
                source[batch * 4 + source_index] * (record.semantic_index == 7 ? 1 : batch + 2);
          }
        }
      }
      for (int repeat = 0; repeat < 2; ++repeat) {
        std::vector<float> output(batches * 5 + 2, 7);
        uint64_t binds = 0;
        {
          testing::ThreadPool pool(1);
          assert(!testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
              pool.get(), planned, source.data(), output.data() + 1, 4, 5, batches,
              [&](std::size_t index, uint64_t batch, auto execute) {
                assert(index == 7 || index == 9);
                assert(index == (binds % 2 == 0 ? 7 : 9));
                assert(batch == binds / 2);
                ++binds;
                if (index == 7) execute(expression::Identity<scalar::F32>{});
                else execute(expression::Scale<scalar::S32, expression::Identity<scalar::F32>>{
                    static_cast<int32_t>(batch + 2), {}});
              })).failure());
        }
        assert(binds == 2 * batches);
        assert(output == expected);
        assert(source == original);
      }
    }
  }
  float output[] = {7, 7, 7};
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {}, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
    ExecuteReductionBatch<scalar::F32, scalar::F32>(
        programs, nullptr, output, 3, [](std::size_t, auto) { assert(false); });
  }
  assert(output[0] == 0 && output[1] == 0 && output[2] == 0);
  {
    testing::ThreadPool pool(1);
    assert(!testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), {}, nullptr, nullptr, 0, 0, 3, [](std::size_t, uint64_t, auto) { assert(false); })).failure());
  }
  const auto empty = layout::BuildLayout({0}, {INT64_MIN}, 0, {INT64_MAX}, 3, 0, 3, 0);
  std::fill_n(output, 3, 7);
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {empty}, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F32>), false);
    ExecuteReductionBatch<scalar::F32, scalar::F32>(
        programs, nullptr, output, 3, [](std::size_t, auto) { assert(false); });
  }
  assert(output[0] == 0 && output[1] == 0 && output[2] == 0);
}

template <typename Function>
void Rejects(Function function) {
  bool rejected = false;
  try { function(); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

void CheckConstruction() {
  const auto overlap = layout::BuildLayout({2, 2}, {1, 2}, 0, {1, 1}, 0, 4, 3, 0);
  Rejects([&] { layout::ValidateInjectiveView(overlap, false, {0, 1}); });
  Rejects([] { layout::BuildReductionLayout({2, 2}, {1, 2}, 0, {2, 2}, {1, 1}, 0,
      {false, false}, 4, 3, 0); });
  const auto dot = layout::BuildLayout({2, 2}, {1, 1}, 0, {1, 1}, 0, 3, 3, 0);
  assert(layout::ElementCount(dot) == 4);
  const std::vector<layout::Record> invalid{
      {0, 0, 0, {2}, {}, {1}},
      {0, 0, 0, {2}, {1}, {}},
      {0, -1, 0, {0}, {1}, {1}},
      {0, 0, -1, {0}, {1}, {1}},
      {0, 65, 0, {0}, {1}, {1}},
      {0, 0, 65, {0}, {1}, {1}},
      {0, 63, 0, {2}, {1}, {1}},
      {0, 0, 63, {2}, {1}, {1}},
      {0, 0, 0, {2}, {-1}, {1}},
      {0, 0, 0, {2}, {1}, {-1}},
      {0, 0, 0, {3}, {INT64_MAX}, {0}},
      {0, 0, 0, {3}, {0}, {INT64_MIN}},
      {0, 0, 0, {UINT64_MAX, 2}, {0, 0}, {0, 0}}};
  for (const auto& record : invalid) {
    Rejects([&] { layout::BuildLayout(record.shape, record.source_strides,
        record.source_offset, record.destination_strides, record.destination_offset,
        64, 64, record.semantic_index); });
  }
  Rejects([] { layout::BuildLayout({1}, {0}, 0, {0}, 0, 0, 1, 0); });
  Rejects([] { layout::BuildLayout({1}, {0}, 0, {0}, 0, 1, 0, 0); });
  float output = 7;
  testing::ThreadPool pool(1);
  for (const auto& sizes : std::array<std::array<uint64_t, 2>, 2>{{{UINT64_MAX, 1}, {0, UINT64_MAX}}}) {
    const auto error = testing::CompletedError(ExecuteReduction<scalar::F32, scalar::F32>(
        pool.get(), {}, nullptr, &output, sizes[0], sizes[1], 2,
        [](std::size_t, uint64_t, auto) { assert(false); }));
    assert(error.failure());
    assert(error.message() == "tensor0-native: reduction batch storage exceeds address range");
    assert(output == 7);
  }
}

void CheckNumeric() {
  const layout::Record overlap{0, 0, 0, {2, 2}, {1, 2}, {1, 1}};
  const double source[] = {0, 1 + std::ldexp(1., -24), 0, 0};
  float output[] = {0, -1, 0};
  kernels::ExecuteReductionRecord<scalar::F32>(overlap, source, output, expression::Identity<scalar::F64>{});
  assert(output[1] == std::ldexp(1.f, -24));
  output[1] = -1;
  kernels::ExecuteReductionRecord<scalar::F32>(overlap, source, output,
      expression::Cast<scalar::F32, expression::Identity<scalar::F64>>{});
  assert(output[1] == 0);
  const layout::Record local{0, 0, 0, {2, 2, 2}, {1, 2, 4}, {0, 1, 1}};
  const double values[] = {0, 0, 1 + std::ldexp(1., -24), -1, 0, 0, 0, 0};
  float local_output[] = {0, 0, 0};
  kernels::ExecuteReductionRecord<scalar::F32>(local, values, local_output, expression::Identity<scalar::F64>{});
  assert(local_output[0] == 0 && local_output[1] == std::ldexp(1.f, -24));
  const layout::Record ordered{0, 0, 0, {2, 2, 2}, {4, 1, 2}, {1, 1, 1}};
  const float ordered_source[] = {0, -1e20f, 1, 0, 1e20f, 0, 0, 0};
  for (bool optimize : {false, true}) {
    auto planned = ordered;
    if (optimize) layout::OptimizeRecordForExecution(&planned);
    float sums[] = {0, 0, 0, 0};
    kernels::ExecuteReductionRecord<scalar::F32>(planned, ordered_source, sums, expression::Identity<scalar::F32>{});
    assert(sums[1] == (optimize ? 0.f : 1.f));
  }
  const std::complex<float> complex_source[] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  std::complex<float> complex_output[3]{};
  kernels::ExecuteReductionRecord<scalar::C64>(overlap, complex_source, complex_output,
      expression::Conjugate<scalar::C64>{});
  assert(complex_output[1] == std::complex<float>(8, -10));
  const int32_t integer_source[] = {0, INT32_MAX, 1, 0};
  int32_t integer_output[3]{};
  kernels::ExecuteReductionRecord<scalar::S32>(overlap, integer_source, integer_output,
      expression::Identity<scalar::S32>{});
  assert(integer_output[1] == INT32_MIN);
}

int main() {
  CheckConstruction();
  CheckLayouts();
  CheckExecution();
  CheckNumeric();
}
