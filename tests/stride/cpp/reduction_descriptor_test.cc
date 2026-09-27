#include <algorithm>
#include <bit>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"
#include "layout/traversal.h"
#include "layout/blocking.h"
#include "ffi/prepared.h"
#include "execute/scheduling.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "kernels/generic.h"
#include "kernels/specialized.h"
#include "kernels/dispatch.h"
#include "execute/reduction.h"

using namespace tensor0::stride;
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
    assert(decoded.records[index].reduction_axes == inputs[index].reduction_axes);
    const auto& record = decoded.records[index].layout;
    assert(record.semantic_index == index);
    assert(record.shape == expected.shape);
    assert(record.source_offset == expected.source_offset);
    assert(record.destination_offset == expected.destination_offset);
    assert(record.source_strides == expected.source_strides);
    assert(record.destination_strides == expected.destination_strides);
  }
  auto state = InstantiateReduction(
      ffi::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()), {});
  assert(state.has_value());
  const auto& prepared = *state;
  assert(prepared->source_size == decoded.source_size);
  assert(prepared->output_size == decoded.output_size);
  assert(prepared->records.size() == inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    auto expected = Build(inputs[index], 7, 5);
    layout::ReductionRecord normalized{expected, inputs[index].reduction_axes};
    layout::OptimizeRecordForPreparation(normalized);
    expected = std::move(normalized.layout);
    const auto& record = prepared->records[index];
    assert(record.semantic_index == expected.semantic_index);
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
          pool.get(), prepared->records, source.data(), result.data() + 1, prepared->source_size,
          prepared->output_size, 2,
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
    std::vector<layout::Record> records;
    for (const auto& record : decoded.records) records.push_back(record.layout);
    const auto programs = layout::PrepareGeneratedRecords(
        records, sizeof(scalar::Value<scalar::S32>), sizeof(scalar::Value<scalar::S32>), false);
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
  assert(huge_decoded.records[0].layout.shape[0] == UINT64_MAX);
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
             .records[0].layout.shape == high_rank.shape);
  const ReductionInput overflow{0, 0, 0, {UINT64_MAX, 2}, {0, 0}, {1, 1}, {1, 1}, {true, true}};
  ExpectInvalid(Bytes(Encode({overflow}, 1, 1)));
}

void CheckMapPreparation() {
  const std::vector<int64_t> words{
      1, 6, 6, 1, 3, 0, 0,
      2, 1, 3, 3, INT64_MIN, 1, 3, INT64_MAX, 1};
  const auto decoded = descriptor::DecodeLayout(words.data(), words.size());
  assert((decoded.records[0].shape == std::vector<uint64_t>{2, 1, 3}));
  auto expected = decoded.records[0];
  layout::OptimizeRecordForPreparation(expected);
  assert((expected.shape == std::vector<uint64_t>{6}));
  auto state = InstantiatePrepared(ffi::Span<const int64_t>(words.data(), words.size()));
  assert(state.has_value());
  const auto& record = (*state)->records[0];
  assert(record.semantic_index == expected.semantic_index);
  assert(record.shape == expected.shape);
  assert(record.source_offset == expected.source_offset);
  assert(record.destination_offset == expected.destination_offset);
  assert(record.source_strides == expected.source_strides);
  assert(record.destination_strides == expected.destination_strides);
}

struct BridgeResult {
  int calls = 0;
  int32_t status = -1;
  std::vector<int64_t> words;
  std::string error;
};

extern "C" void ReceivePrepared(
    void* context, int32_t status, const int64_t* words, std::size_t count,
    const char* error, std::size_t error_size) noexcept {
  auto& result = *static_cast<BridgeResult*>(context);
  ++result.calls;
  result.status = status;
  if (status == 0) {
    assert(error == nullptr && error_size == 0);
    result.words.assign(words, words + count);
  } else {
    assert(words == nullptr && count == 0 && error != nullptr);
    result.error.assign(error, error_size);
  }
}

BridgeResult Bridge(int32_t operation, const std::vector<int64_t>& words) {
  BridgeResult result;
  Tensor0StridePrepareLayout(operation, words.data(), words.size(), &result, ReceivePrepared);
  assert(result.calls == 1);
  return result;
}

void SameRecord(const layout::Record& left, const layout::Record& right) {
  assert(left.semantic_index == right.semantic_index);
  assert(left.source_offset == right.source_offset);
  assert(left.destination_offset == right.destination_offset);
  assert(left.shape == right.shape);
  assert(left.source_strides == right.source_strides);
  assert(left.destination_strides == right.destination_strides);
}

void SameState(const PreparedState& state, const descriptor::DecodedLayout& decoded) {
  assert(state.source_size == decoded.source_size);
  assert(state.output_size == decoded.output_size);
  assert(state.records.size() == decoded.records.size());
  for (std::size_t i = 0; i < state.records.size(); ++i) {
    assert(state.records[i].semantic_index == i);
    SameRecord(state.records[i], decoded.records[i]);
  }
}

void SameState(const PreparedState& left, const PreparedState& right) {
  assert(left.source_size == right.source_size);
  assert(left.output_size == right.output_size);
  assert(left.records.size() == right.records.size());
  for (std::size_t i = 0; i < left.records.size(); ++i)
    SameRecord(left.records[i], right.records[i]);
}

std::vector<int64_t> SignedWords(const std::vector<uint64_t>& words) {
  std::vector<int64_t> signed_words;
  for (auto word : words) signed_words.push_back(std::bit_cast<int64_t>(word));
  return signed_words;
}

// The bridge emits a little-endian unsigned reduction protocol, carried in signed words.
std::string ReductionBytes(const std::vector<int64_t>& words) {
  std::vector<uint64_t> unsigned_words;
  for (auto word : words) unsigned_words.push_back(std::bit_cast<uint64_t>(word));
  return Bytes(unsigned_words);
}

void CheckSharedPreparation() {
  // Multiple records preserve semantic indices (and therefore coefficient slots).
  // Nonempty singleton axes are removed after stable sorting and fusion.
  const std::vector<int64_t> map{
      1, 32, 32, 4,
      3, 0, 0, 2, 3, 1, 3, 1, INT64_MIN, 3, 1, INT64_MAX,
      0, 4, 12,
      2, 9, 16, 2, 2, -2, -1, 2, 1,
      2, 0, 0, 0, 2, 9, 1, 9, 1};
  const auto original = descriptor::DecodeLayout(map.data(), map.size());
  assert((original.records[0].shape == std::vector<uint64_t>{2, 3, 1}));
  for (int32_t operation : {0, 1}) {
    const auto bridge = Bridge(operation, map);
    assert(bridge.status == 0);
    const auto decoded = descriptor::DecodeLayout(bridge.words.data(), bridge.words.size());
    assert((decoded.records[0].shape == std::vector<uint64_t>{6}));
    assert((decoded.records[2].shape == std::vector<uint64_t>{4}));
    assert((decoded.records[3].shape == std::vector<uint64_t>{0, 2}));
    const auto raw = InstantiatePrepared(ffi::Span<const int64_t>(map.data(), map.size()));
    const auto already = InstantiatePrepared(
        ffi::Span<const int64_t>(bridge.words.data(), bridge.words.size()));
    assert(raw.has_value() && already.has_value());
    SameState(**raw, decoded);
    SameState(**raw, **already);
    const auto twice = Bridge(operation, bridge.words);
    assert(twice.status == 0 && twice.words == bridge.words);
  }

  // Non-injective repeated destinations are legal for dot/accumulation, not map.
  const std::vector<int64_t> address{
      1, 32, 32, 3,
      2, 0, 0, 2, 3, 3, 1, 0, 0,
      2, 5, 5, 2, 2, -2, -1, -2, -1,
      0, 4, 5};
  assert(Bridge(0, address).status == 1);
  for (int32_t operation : {2, 3}) {
    const auto bridge = Bridge(operation, address);
    assert(bridge.status == 0);
    const auto decoded = descriptor::DecodeAddressLayout(bridge.words.data(), bridge.words.size());
    assert((decoded.records[0].shape == std::vector<uint64_t>{6}));
    assert((decoded.records[1].shape == std::vector<uint64_t>{4}));
    const auto instantiate = [&](const std::vector<int64_t>& words) {
      const auto span = ffi::Span<const int64_t>(words.data(), words.size());
      return operation == 2 ? InstantiateDot(span, 0)
                            : InstantiateAccumulation(span, {});
    };
    const auto raw = instantiate(address);
    const auto already = instantiate(bridge.words);
    assert(raw.has_value() && already.has_value());
    SameState(**raw, decoded);
    SameState(**raw, **already);
    const auto twice = Bridge(operation, bridge.words);
    assert(twice.status == 0 && twice.words == bridge.words);
  }

  const std::vector<ReductionInput> inputs{
      {0, 0, 0, {2, 3, 2, 2, 1}, {12, 4, 2, 1, INT64_MIN},
       {2, 3, 1, 1, 1}, {3, 1, INT64_MAX, INT64_MIN, INT64_MAX},
       {false, false, true, true, false}},
      {1, 5, 5, {2, 2}, {-2, -1}, {2, 2}, {-2, -1}, {false, false}},
      {2, 4, 4, {}, {}, {}, {}, {}},
      {3, 0, 0, {0, 2}, {INT64_MIN, 1}, {1, 2}, {INT64_MAX, 1}, {true, false}}};
  const auto reduction = SignedWords(Encode(inputs, 24, 12));
  const auto bridge = Bridge(4, reduction);
  assert(bridge.status == 0);
  const auto bytes = ReductionBytes(bridge.words);
  const auto decoded = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  assert(layout::ElementCount(decoded.records[0].layout) == 24);
  assert(decoded.records[0].reduction_axes.size() == decoded.records[0].layout.shape.size());
  for (std::size_t axis = 0; axis < decoded.records[0].reduction_axes.size(); ++axis)
    if (decoded.records[0].reduction_axes[axis])
      assert(decoded.records[0].layout.destination_strides[axis] == 0);
  assert((decoded.records[1].layout.shape == std::vector<uint64_t>{4}));
  const auto instantiate_reduction = [&](const std::vector<int64_t>& words) {
    const auto data = ReductionBytes(words);
    return InstantiateReduction(ffi::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()), {});
  };
  const auto raw = instantiate_reduction(reduction);
  const auto already = instantiate_reduction(bridge.words);
  assert(raw.has_value() && already.has_value());
  descriptor::DecodedLayout projected;
  projected.source_size = decoded.source_size;
  projected.output_size = decoded.output_size;
  for (const auto& record : decoded.records) projected.records.push_back(record.layout);
  SameState(**raw, projected);
  SameState(**raw, **already);
  const auto twice = Bridge(4, bridge.words);
  assert(twice.status == 0 && twice.words == bridge.words);

  // Unsigned extents are legal even when they exceed the signed map protocol.
  // A contiguous same-role pair fuses all the way to UINT64_MAX.
  const std::vector<ReductionInput> huge_inputs{
      {0, 0, 0, {UINT64_MAX / 3, 3}, {0, 0}, {1, 1}, {INT64_MIN, INT64_MAX},
       {true, true}}};
  const auto huge_words = SignedWords(Encode(huge_inputs, 1, 1));
  const auto huge_bridge = Bridge(4, huge_words);
  assert(huge_bridge.status == 0);
  const auto huge_bytes = ReductionBytes(huge_bridge.words);
  const auto huge_decoded = descriptor::DecodeReductionLayout(
      huge_bytes.data(), huge_bytes.size());
  assert((huge_decoded.records[0].layout.shape == std::vector<uint64_t>{UINT64_MAX}));
  assert((huge_decoded.records[0].reduction_axes == std::vector<bool>{true}));
  const auto huge_raw = instantiate_reduction(huge_words);
  const auto huge_already = instantiate_reduction(huge_bridge.words);
  assert(huge_raw.has_value() && huge_already.has_value());
  SameRecord((*huge_raw)->records[0], huge_decoded.records[0].layout);
  SameState(**huge_raw, **huge_already);
  const auto huge_twice = Bridge(4, huge_bridge.words);
  assert(huge_twice.status == 0 && huge_twice.words == huge_bridge.words);

  for (int32_t operation = 0; operation < 5; ++operation) {
    const std::vector<int64_t> empty{1, 0, 0, 0};
    const auto prepared = Bridge(operation, empty);
    assert(prepared.status == 0 && prepared.words == empty);
    if (operation == 4) {
      const auto state = instantiate_reduction(empty);
      assert(state.has_value() && (*state)->records.empty());
    } else {
      const auto state = operation < 2
          ? InstantiatePrepared(ffi::Span<const int64_t>(empty.data(), empty.size()))
          : operation == 2
              ? InstantiateDot(ffi::Span<const int64_t>(empty.data(), empty.size()), 0)
              : InstantiateAccumulation(ffi::Span<const int64_t>(empty.data(), empty.size()), {});
      assert(state.has_value() && (*state)->records.empty());
    }
  }

  // Complete validation precedes fusion; malformed later records/trailing words
  // must not be hidden by a fusible first record.
  for (int32_t operation = 0; operation < 5; ++operation) {
    auto invalid = operation == 4 ? reduction : (operation < 2 ? map : address);
    invalid.push_back(99);
    const auto rejected = Bridge(operation, invalid);
    assert(rejected.status == 1 && !rejected.error.empty());
    invalid.pop_back();
    // The first record is fusible; an invalid offset in a subsequent record
    // must still be rejected before any prepared descriptor is returned.
    invalid[operation == 4 ? 49 : operation < 2 ? 17 : 23] = -1;
    const auto later = Bridge(operation, invalid);
    assert(later.status == 1 && !later.error.empty());
  }
  auto overflow = SignedWords(Encode({
      {0, 0, 0, {UINT64_MAX, 2, 0}, {0, 0, 0}, {1, 2, 1}, {0, 1, 0},
       {true, false, true}},
      {1, 0, 0, {}, {}, {}, {}, {}}}, 0, 2));
  overflow[overflow.size() - 2] = -1;  // Invalid later record offset.
  overflow.push_back(99);             // Also invalid trailing data.
  const auto rejected = Bridge(4, overflow);
  assert(rejected.status == 1 && rejected.error == "layout element count overflows");
  const auto overflow_bytes = ReductionBytes(overflow);
  bool same_error = false;
  try {
    descriptor::PrepareReductionLayout(overflow_bytes.data(), overflow_bytes.size());
  } catch (const std::invalid_argument& error) {
    same_error = rejected.error == error.what();
  }
  assert(same_error);
}

void CheckSingletonExecution() {
  for (int64_t stride : {INT64_MAX, INT64_MIN}) {
    const std::vector<int64_t> words{1, 2, 2, 1, 1, 1, 1, 1, stride, stride};
    const auto bridge = Bridge(0, words);
    assert(bridge.status == 0 &&
           bridge.words == (std::vector<int64_t>{1, 2, 2, 1, 0, 1, 1}));
    const auto raw = descriptor::DecodeLayout(words.data(), words.size()).records[0];
    const auto state = InstantiatePrepared(ffi::Span<const int64_t>(words.data(), words.size()));
    assert(state.has_value());
    const auto& record = (*state)->records[0];
    const int32_t source[2]{11, 37};
    int32_t result[2]{19, 23};
    kernels::ExecuteGenericAffineScalarRow(
        source, result, raw.source_offset, raw.destination_offset,
        raw.shape[0], raw.source_strides[0], raw.destination_strides[0],
        expression::Identity<scalar::S32>{});
    assert(result[0] == 19 && result[1] == 37);
    result[1] = 23;
    kernels::ExecuteBinaryMapRecord<scalar::S32>(
        record, source, result, result,
        expression::AddUpdate<expression::Identity<scalar::S32>,
                              expression::Identity<scalar::S32>>{});
    assert(result[0] == 19 && result[1] == 60);
    assert(record.shape.empty());
    auto same_addresses = raw;
    same_addresses.destination_strides[0] = stride == INT64_MAX ? INT64_MIN : INT64_MAX;
    assert(layout::HasIdenticalAddresses(same_addresses));
    same_addresses.destination_offset = 0;
    assert(!layout::HasIdenticalAddresses(same_addresses));
  }
  // Empty records preserve the historical exact-stride-vector alias policy.
  const std::vector<int64_t> empty{1, 2, 2, 1, 2, 0, 0, 0, 1,
                                   INT64_MAX, 1, INT64_MAX, INT64_MIN};
  const auto empty_state = InstantiatePrepared(
      ffi::Span<const int64_t>(empty.data(), empty.size()));
  assert(empty_state.has_value());
  assert(!layout::HasIdenticalAddresses((*empty_state)->records[0]));
  // Same address bounds do not make a nontrivial transpose pointwise identical.
  const std::vector<int64_t> transpose{1, 4, 4, 1, 2, 0, 0, 2, 2, 2, 1, 1, 2};
  const auto transpose_state = InstantiatePrepared(
      ffi::Span<const int64_t>(transpose.data(), transpose.size()));
  assert(transpose_state.has_value());
  assert(!layout::HasIdenticalAddresses((*transpose_state)->records[0]));
}

int main() {
  CheckSharedPreparation();
  CheckSingletonExecution();
  CheckMapPreparation();
  CheckExecution();
  CheckBoundary();
}
