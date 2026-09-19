#include <atomic>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
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
#include "reduction_input.inc"

std::string Bytes(const std::vector<uint64_t>& words) {
  std::string bytes;
  for (uint64_t word : words) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      bytes.push_back(static_cast<char>((word >> shift) & 255));
    }
  }
  return bytes;
}

std::vector<uint64_t> Encode(const std::vector<ReductionInput>& inputs,
                             uint64_t source_size, uint64_t output_size) {
  std::vector<uint64_t> words{1, source_size, output_size, inputs.size()};
  for (const auto& input : inputs) {
    words.insert(words.end(), {input.shape.size(), static_cast<uint64_t>(input.source_offset),
                              static_cast<uint64_t>(input.destination_offset)});
    words.insert(words.end(), input.shape.begin(), input.shape.end());
    for (auto stride : input.source_strides) words.push_back(static_cast<uint64_t>(stride));
    words.insert(words.end(), input.output_shape.begin(), input.output_shape.end());
    for (auto stride : input.destination_strides) words.push_back(static_cast<uint64_t>(stride));
    for (bool flag : input.reduction_axes) words.push_back(flag);
  }
  return words;
}

void ExpectInvalid(const std::string& bytes) {
  bool rejected = false;
  try {
    descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void CheckExecution() {
  const std::vector<ReductionInput> inputs{
      {0, 0, 1, {2, 2}, {2, 1}, {2, 1}, {2, INT64_MAX}, {false, true}},
      {1, 5, 3, {2}, {-1}, {2}, {-2}, {false}},
      {2, 6, 1, {}, {}, {}, {}, {}},
      {3, 7, 4, {0}, {1}, {1}, {INT64_MAX}, {true}}};
  const auto bytes = Bytes(Encode(inputs, 7, 5));
  const auto decoded = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  assert(decoded.source_size == 7 && decoded.output_size == 5);
  assert(decoded.records.size() == inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto expected = Build(inputs[index], 7, 5);
    layout::OptimizeRecordForExecution(&expected);
    const auto& record = decoded.records[index];
    assert(record.semantic_index == index);
    assert(record.shape == expected.shape);
    assert(record.source_offset == expected.source_offset);
    assert(record.destination_offset == expected.destination_offset);
    assert(record.source_strides == expected.source_strides);
    assert(record.destination_strides == expected.destination_strides);
  }
  const std::array<int32_t, 14> source{1, 2, 3, 4, 5, 6, 7, 2, 4, 6, 8, 10, 12, 14};
  std::array<int32_t, 12> result;
  result.fill(99);
  for (int32_t factor : {1, 2}) {
    std::vector<std::size_t> bound;
    {
      testing::ThreadPool pool(1);
      assert(!testing::CompletedError(ExecuteReduction<scalar::S32, scalar::S32>(
          pool.get(), decoded.records, source.data(), result.data() + 1, decoded.source_size,
          decoded.output_size, 2,
          [&](std::size_t index, uint64_t, auto execute) {
            bound.push_back(index);
            execute(expression::Scale<scalar::S32>{factor});
          })).failure());
    }
    assert((bound == std::vector<std::size_t>{0, 1, 2, 0, 1, 2}));
    assert((result == std::array<int32_t, 12>{
        99, 0, 15 * factor, 0, 13 * factor, 0,
        0, 30 * factor, 0, 26 * factor, 0, 99}));
  }
  const auto unaligned = std::string(1, '\0') + bytes;
  const auto decoded_unaligned = descriptor::DecodeReductionLayout(
      unaligned.data() + 1, bytes.size());
  assert(decoded_unaligned.records.size() == inputs.size());
  for (std::size_t length = 0; length < bytes.size(); ++length) {
    ExpectInvalid(bytes.substr(0, length));
  }
  ExpectInvalid(bytes + std::string(8, '\0'));
}

void CheckBoundary() {
  const auto empty = Bytes({1, 0, 3, 0});
  const auto decoded = descriptor::DecodeReductionLayout(empty.data(), empty.size());
  std::array<int32_t, 3> result{7, 7, 7};
  {
    const auto programs = layout::PrepareGeneratedRecords(
        decoded.records, sizeof(scalar::Value<scalar::S32>), sizeof(scalar::Value<scalar::S32>), false);
    ExecuteReductionBatch<scalar::S32, scalar::S32>(
        programs, nullptr, result.data(), decoded.output_size,
        [](std::size_t, auto execute) {
          assert(false);
          execute(expression::Identity<scalar::S32>{});
        });
  }
  assert((result == std::array<int32_t, 3>{0, 0, 0}));
  const ReductionInput broadcast{0, 0, 0, {UINT64_MAX}, {0}, {1}, {INT64_MIN}, {true}};
  const auto huge = Bytes(Encode({broadcast}, 1, 1));
  const auto huge_decoded = descriptor::DecodeReductionLayout(huge.data(), huge.size());
  assert(huge_decoded.records[0].shape[0] == UINT64_MAX);
  const auto valid = Encode({{0, 2, 2, {2}, {-1}, {2}, {-1}, {false}}}, 3, 3);
  for (const auto& change : std::vector<std::array<uint64_t, 2>>{
           {0, 2}, {3, UINT64_MAX}, {4, UINT64_MAX}, {5, UINT64_MAX},
           {6, UINT64_C(1) << 63}, {7, UINT64_MAX}, {8, UINT64_C(1) << 63},
           {9, 3}, {10, 0}, {11, 2}}) {
    auto words = valid;
    words[change[0]] = change[1];
    ExpectInvalid(Bytes(words));
  }
  const ReductionInput no_source{0, 0, 2, {0}, {1}, {1}, {INT64_MAX}, {true}};
  const auto no_source_bytes = Bytes(Encode({no_source}, 0, 3));
  assert(descriptor::DecodeReductionLayout(no_source_bytes.data(), no_source_bytes.size())
             .records.size() == 1);
  ExpectInvalid(Bytes(Encode({no_source}, 0, 2)));
  const ReductionInput high_rank{0, 0, 0, std::vector<uint64_t>(12, 1),
      std::vector<int64_t>(12, INT64_MIN), std::vector<uint64_t>(12, 1),
      std::vector<int64_t>(12, INT64_MAX), std::vector<bool>(12, true)};
  const auto high_rank_bytes = Bytes(Encode({high_rank}, 1, 1));
  assert(descriptor::DecodeReductionLayout(high_rank_bytes.data(), high_rank_bytes.size())
             .records[0].shape.empty());
  const ReductionInput overflow{0, 0, 0, {UINT64_MAX, 2}, {0, 0}, {1, 1}, {1, 1}, {true, true}};
  ExpectInvalid(Bytes(Encode({overflow}, 1, 1)));
}

int main() {
  CheckExecution();
  CheckBoundary();
}
