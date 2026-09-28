#include <algorithm>
#include <random>
#include "layout/descriptor.h"
#include "layout/owner_fiber.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace tensor0::stride;

std::string Bytes(const std::vector<uint64_t>& words) {
  std::string bytes;
  for (uint64_t word : words) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      bytes.push_back(static_cast<char>((word >> shift) & 255));
    }
  }
  return bytes;
}

template <class Function>
void ExpectInvalid(Function decode) {
  bool rejected = false;
  try {
    decode();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void CheckLayout() {
  const std::vector<int64_t> words{
      1, 6, 6, 1, 3, 0, 0,
      2, 1, 3, 3, INT64_MIN, 1, 3, INT64_MAX, 1};
  const auto decoded = descriptor::DecodeLayout(words.data(), words.size());
  assert(decoded.source_size == 6 && decoded.output_size == 6);
  assert(decoded.records.size() == 1);
  const auto& record = decoded.records[0];
  assert(record.semantic_index == 0);
  assert(record.source_offset == 0 && record.destination_offset == 0);
  assert((record.shape == std::vector<uint64_t>{2, 1, 3}));
  assert((record.source_strides == std::vector<int64_t>{3, INT64_MIN, 1}));
  assert((record.destination_strides == std::vector<int64_t>{3, INT64_MAX, 1}));

  const std::vector<int64_t> collision{1, 4, 3, 1, 2, 0, 0, 2, 2, 2, 1, 1, 1};
  const auto address = descriptor::DecodeAddressLayout(collision.data(), collision.size());
  assert((address.records[0].shape == std::vector<uint64_t>{2, 2}));
  assert((address.records[0].destination_strides == std::vector<int64_t>{1, 1}));
  ExpectInvalid([&] { descriptor::DecodeLayout(collision.data(), collision.size()); });
}

void CheckReduction() {
  const auto bytes = Bytes({
      1, 6, 2, 1, 3, 0, 0,
      2, 1, 3, 3, UINT64_C(1) << 63, 1,
      2, 1, 1, 1, INT64_MAX, UINT64_C(1) << 63, 0, 0, 1});
  const auto decoded = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  assert(decoded.source_size == 6 && decoded.output_size == 2);
  assert(decoded.records.size() == 1);
  const auto expected = layout::BuildReductionLayout(
      {2, 1, 3}, {3, INT64_MIN, 1}, 0, {2, 1, 1}, {1, INT64_MAX, INT64_MIN},
      0, {false, false, true}, 6, 2, 0);
  assert((decoded.records[0].reduction_axes == std::vector<bool>{false, false, true}));
  const auto& record = decoded.records[0].layout;
  assert(record.semantic_index == expected.semantic_index);
  assert(record.source_offset == expected.source_offset);
  assert(record.destination_offset == expected.destination_offset);
  assert(record.shape == expected.shape);
  assert(record.source_strides == expected.source_strides);
  assert(record.destination_strides == expected.destination_strides);
  assert((record.shape == std::vector<uint64_t>{2, 1, 3}));
  assert((record.destination_strides == std::vector<int64_t>{1, INT64_MAX, 0}));

  // An empty reduction fiber still requires an injective output map.
  const auto empty_fiber = Bytes({
      1, 0, 3, 1, 3, 0, 0,
      2, 2, 0, 0, 0, 0, 2, 2, 1, 1, 1, INT64_MAX, 0, 0, 1});
  ExpectInvalid([&] {
    descriptor::DecodeReductionLayout(empty_fiber.data(), empty_fiber.size());
  });

  // Grouped map/fiber products can be valid even when the original-axis prefix
  // overflows before a zero extent. Preserve the original rejection priority.
  const auto prefix_overflow = Bytes({
      1, 0, 2, 1, 3, 0, 0,
      UINT64_MAX, 2, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1});
  auto before_invalid_record = prefix_overflow;
  before_invalid_record[24] = 2;  // Two records in the V1 header.
  before_invalid_record += Bytes({0, UINT64_MAX, 0});
  for (const auto& invalid : {prefix_overflow, prefix_overflow + std::string(8, '\0'),
                              before_invalid_record}) {
    bool rejected = false;
    try {
      descriptor::DecodeReductionLayout(invalid.data(), invalid.size());
    } catch (const std::invalid_argument& error) {
      assert(std::string(error.what()) == "layout element count overflows");
      rejected = true;
    }
    assert(rejected);
  }

  // Reduction extents are unsigned, including values beyond INT64_MAX.
  auto huge_words = std::vector<uint64_t>{
      1, 1, 1, 1, 1, 0, 0, UINT64_MAX, 0, 1, UINT64_C(1) << 63, 1};
  const auto huge = Bytes(huge_words);
  const auto huge_decoded = descriptor::DecodeReductionLayout(huge.data(), huge.size());
  assert((huge_decoded.records[0].layout.shape == std::vector<uint64_t>{UINT64_MAX}));
  assert((huge_decoded.records[0].layout.destination_strides == std::vector<int64_t>{0}));
  huge_words[5] = UINT64_C(1) << 63;
  const auto invalid_offset = Bytes(huge_words);
  ExpectInvalid([&] {
    descriptor::DecodeReductionLayout(invalid_offset.data(), invalid_offset.size());
  });
}

void SameRecord(const layout::Record& left, const layout::Record& right) {
  assert(left.semantic_index == right.semantic_index);
  assert(left.source_offset == right.source_offset);
  assert(left.destination_offset == right.destination_offset);
  assert(left.shape == right.shape);
  assert(left.source_strides == right.source_strides);
  assert(left.destination_strides == right.destination_strides);
}

void RoundTripLayout(const std::vector<int64_t>& words, bool map = true) {
  const auto decode = map ? descriptor::DecodeLayout : descriptor::DecodeAddressLayout;
  const auto original = decode(words.data(), words.size());
  const auto encoded = descriptor::EncodeLayout(original);
  assert(encoded == words);
  const auto roundtrip = decode(encoded.data(), encoded.size());
  assert(roundtrip.source_size == original.source_size);
  assert(roundtrip.output_size == original.output_size);
  assert(roundtrip.records.size() == original.records.size());
  for (std::size_t i = 0; i < original.records.size(); ++i) {
    assert(original.records[i].semantic_index == i);
    SameRecord(original.records[i], roundtrip.records[i]);
  }
  assert(descriptor::EncodeLayout(roundtrip) == encoded);
}

void RoundTripReduction(const std::vector<uint64_t>& words,
                        const std::vector<uint64_t>& canonical) {
  const auto bytes = Bytes(words);
  const auto original = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  const auto encoded = descriptor::EncodeReductionLayout(original);
  // Compare actual little-endian bytes, including unsigned sizes and stride bits.
  assert(std::string(encoded.begin(), encoded.end()) == Bytes(canonical));
  const auto roundtrip = descriptor::DecodeReductionLayout(encoded.data(), encoded.size());
  assert(roundtrip.source_size == original.source_size);
  assert(roundtrip.output_size == original.output_size);
  assert(roundtrip.records.size() == original.records.size());
  for (std::size_t i = 0; i < original.records.size(); ++i) {
    assert(original.records[i].layout.semantic_index == i);
    SameRecord(original.records[i].layout, roundtrip.records[i].layout);
    assert(original.records[i].reduction_axes == roundtrip.records[i].reduction_axes);
  }
  assert(descriptor::EncodeReductionLayout(roundtrip) == encoded);
}

void CheckRoundTrips() {
  RoundTripLayout({1, 0, 0, 0});
  RoundTripLayout({1, INT64_MAX, INT64_MAX, 0});
  // Rank-zero and empty records keep their offsets and order.
  RoundTripLayout({1, 8, 9, 3, 0, 3, 4, 1, 8, 9, 0, INT64_MIN, INT64_MAX,
                   2, 5, 6, 2, 1, -2, INT64_MIN, -3, 0});
  RoundTripLayout({1, 4, 3, 1, 2, 0, 0, 2, 2, 2, 1, 1, 1}, false);
  RoundTripLayout({1, 1, 1, 1, 1, 0, 0, 3, 0, 0}, false);

  const std::vector<uint64_t> empty{1, UINT64_MAX, UINT64_C(1) << 63, 0};
  RoundTripReduction(empty, empty);
  const std::vector<uint64_t> scalar{1, 8, 9, 2, 0, 3, 4, 0, 2, 1};
  RoundTripReduction(scalar, scalar);
  // Identical singleton shapes and zero destination strides do not imply roles.
  const std::vector<uint64_t> singletons{
      1, 1, 1, 1, 2, 0, 0, 1, 1, UINT64_C(1) << 63, INT64_MAX,
      1, 1, 0, 0, 0, 1};
  RoundTripReduction(singletons, singletons);
  const std::vector<uint64_t> huge{
      1, UINT64_MAX, UINT64_C(1) << 63, 1, 1, INT64_MAX, INT64_MAX,
      UINT64_MAX, 0, 1, UINT64_C(1) << 63, 1};
  auto canonical_huge = huge;
  canonical_huge[10] = 0;
  RoundTripReduction(huge, canonical_huge);
  const auto negative = static_cast<uint64_t>(-2);
  const std::vector<uint64_t> mixed{
      1, 8, 9, 3,
      2, 5, 6, 2, 1, negative, UINT64_C(1) << 63, 2, 1, negative, INT64_MAX, 0, 1,
      1, 8, 4, 0, UINT64_C(1) << 63, 1, INT64_MAX, 1,
      1, 8, 9, 0, INT64_MAX, 0, UINT64_C(1) << 63, 0};
  auto canonical_mixed = mixed;
  canonical_mixed[14] = 0;
  canonical_mixed[23] = 0;
  RoundTripReduction(mixed, canonical_mixed);
}

struct BridgeResult {
  int calls = 0;
  int32_t status = -1;
  std::vector<int64_t> words;
  std::string error;
};

extern "C" void ReceiveLayout(
    void* context, int32_t status, const int64_t* words, std::size_t word_count,
    const char* error, std::size_t error_size) noexcept {
  auto& result = *static_cast<BridgeResult*>(context);
  ++result.calls;
  result.status = status;
  if (status == 0) {
    assert(error == nullptr && error_size == 0);
    if (word_count != 0) result.words.assign(words, words + word_count);
  } else {
    assert(words == nullptr && word_count == 0);
    assert(error != nullptr && error_size != 0);
    result.error.assign(error, error_size);
  }
}

BridgeResult PrepareLayout(int32_t operation, const std::vector<int64_t>& words) {
  BridgeResult result;
  Tensor0StridePrepareLayout(operation, words.data(), words.size(), &result, ReceiveLayout);
  assert(result.calls == 1);
  return result;
}

void CheckPreparationBridge() {
  const std::vector<int64_t> map{1, 6, 6, 1, 3, 0, 0,
      2, 1, 3, 3, INT64_MIN, 1, 3, INT64_MAX, 1};
  const std::vector<int64_t> collision{1, 4, 3, 1, 2, 0, 0, 2, 2, 2, 1, 1, 1};
  for (int32_t operation = 0; operation < 4; ++operation) {
    const auto copied = PrepareLayout(operation, map);
    assert(copied.status == 0);
    const auto canonical = operation < 2 ? descriptor::PrepareLayout(map.data(), map.size())
                                         : operation == 3 ? descriptor::PrepareAccumulationLayout(map.data(), map.size())
                                                          : descriptor::PrepareAddressLayout(map.data(), map.size());
    assert(copied.words == descriptor::EncodeLayout(canonical));
    assert(PrepareLayout(operation, copied.words).words == copied.words);
    const auto address = PrepareLayout(operation, collision);
    assert(address.status == (operation == 2 ? 0 : 1));
    if (operation == 2) {
      assert(address.words == descriptor::EncodeLayout(
          descriptor::PrepareAddressLayout(collision.data(), collision.size())));
    }
  }

  // Signed words transport unsigned reduction sizes and negative stride bits.
  // The result owns its copy after native temporary buffers have been freed.
  std::vector<int64_t> reduction{
      1, -1, INT64_MIN, 1, 1, INT64_MAX, INT64_MAX, -1, 0, 1, INT64_MIN, 1};
  const auto copied = PrepareLayout(4, reduction);
  reduction[10] = 0;
  assert(copied.status == 0 && copied.words == reduction);
  const auto again = PrepareLayout(4, copied.words);
  assert(again.status == 0 && again.words == copied.words);
  assert(PrepareLayout(0, copied.words).status == 1);

  for (int32_t operation = 0; operation < 5; ++operation) {
    for (const auto& malformed : {std::vector<int64_t>{},
                                  std::vector<int64_t>{2, 0, 0, 0},
                                  std::vector<int64_t>{1, 0, 0, 0, 0}}) {
      const auto invalid = PrepareLayout(operation, malformed);
      assert(invalid.status == 1 && invalid.words.empty() && !invalid.error.empty());
    }
  }
  const auto unknown = PrepareLayout(5, {});
  assert(unknown.status == 1 && unknown.error == "unsupported stride layout operation");
  assert(PrepareLayout(-1, map).status == 1);

  // The original-axis overflow wins over a later invalid record and trailing data.
  std::vector<int64_t> prefix_overflow{
      1, 0, 2, 2, 3, 0, 0, -1, 2, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1,
      0, -1, 0, 0};
  const auto overflow = PrepareLayout(4, prefix_overflow);
  assert(overflow.status == 1 && overflow.error == "layout element count overflows");
}


using Addresses = std::vector<std::pair<int64_t, int64_t>>;

// Enumerate every logical tuple, never merely the set of visited addresses.
void Visit(const layout::Record& record, const std::vector<std::size_t>& axes,
           std::size_t depth, int64_t source, int64_t destination, Addresses& out) {
  if (depth == axes.size()) {
    out.emplace_back(source, destination);
    return;
  }
  const auto axis = axes[depth];
  for (uint64_t i = 0; i < record.shape[axis]; ++i) {
    Visit(record, axes, depth + 1,
          source + static_cast<int64_t>(i) * record.source_strides[axis],
          destination + static_cast<int64_t>(i) * record.destination_strides[axis], out);
  }
}

Addresses Enumerate(const layout::Record& record, const std::vector<bool>* roles = nullptr) {
  std::vector<std::size_t> axes;
  if (roles) {
    // Map tuples outside, fiber tuples inside; role order must survive fusion.
    for (bool role : {false, true})
      for (std::size_t i = 0; i < record.shape.size(); ++i)
        if ((*roles)[i] == role) axes.push_back(i);
  } else {
    for (std::size_t i = 0; i < record.shape.size(); ++i) axes.push_back(i);
  }
  Addresses result;
  Visit(record, axes, 0, record.source_offset, record.destination_offset, result);
  return result;
}

bool AliasProof(const layout::Record& record) {
  if (record.source_offset != record.destination_offset) return false;
  const auto count = layout::ElementCount(record);
  for (std::size_t i = 0; i < record.shape.size(); ++i)
    if ((count == 0 || record.shape[i] > 1) &&
        record.source_strides[i] != record.destination_strides[i]) return false;
  return true;
}

bool InjectivityProof(const layout::Record& record, bool accumulation) {
  std::vector<std::size_t> axes;
  for (std::size_t i = 0; i < record.shape.size(); ++i) {
    // Match Accumulation Classify's map_axes, without CUDA dependencies.
    if (!accumulation || (record.shape[i] > 1 && record.destination_strides[i] != 0))
      axes.push_back(i);
  }
  try {
    layout::ValidateInjectiveView(record, false, axes);
    return true;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

void CheckAddressFusion(const std::vector<int64_t>& words,
                        bool enumerate = true, int32_t operation = 2) {
  const auto before = descriptor::DecodeAddressLayout(words.data(), words.size());
  const auto prepared = PrepareLayout(operation, words);
  assert(prepared.status == 0);
  assert(PrepareLayout(operation, prepared.words).words == prepared.words);
  const auto after = descriptor::DecodeAddressLayout(prepared.words.data(), prepared.words.size());
  assert(before.source_size == after.source_size && before.output_size == after.output_size);
  assert(before.records.size() == after.records.size());
  for (std::size_t i = 0; i < before.records.size(); ++i) {
    const auto& a = before.records[i];
    const auto& b = after.records[i];
    assert(a.semantic_index == i && b.semantic_index == i);
    assert(a.source_offset == b.source_offset && a.destination_offset == b.destination_offset);
    assert(layout::ElementCount(a) == layout::ElementCount(b));
    assert(AliasProof(a) == AliasProof(b));
    for (bool accumulation : {false, true})
      assert(InjectivityProof(a, accumulation) == InjectivityProof(b, accumulation));
    if (enumerate) {
      auto original = Enumerate(a), normalized = Enumerate(b);
      std::sort(original.begin(), original.end());
      std::sort(normalized.begin(), normalized.end());
      assert(original == normalized);
    }
    if (layout::ElementCount(a) == 0) SameRecord(a, b);
  }
}

void CheckFusion() {
  for (int32_t op = 0; op < 4; ++op)
    CheckAddressFusion({1, 24, 24, 1, 3, 0, 0, 2, 3, 4, 12, 4, 1, 12, 4, 1}, true, op);
  // Systematic accepted signed/zero stride pairs, including one-sided contiguity.
  for (int64_t so = -3; so <= 3; ++so)
    for (int64_t si = -3; si <= 3; ++si)
      for (int64_t dO = -3; dO <= 3; ++dO)
        for (int64_t di = -3; di <= 3; ++di) {
          CheckAddressFusion({1, 32, 32, 1, 2, 12, 12, 2, 2, so, si, dO, di});
        }
  CheckAddressFusion({1, 32, 32, 5,
      0, 3, 4,
      3, 0, 0, 2, 1, 3, 3, INT64_MIN, 1, 3, INT64_MAX, 1,
      3, 32, 32, 0, 2, 3, 0, 3, 1, INT64_MIN, 3, 1,
      2, 5, 5, 2, 3, -3, -1, -3, -1,
      3, 0, 0, 2, 3, 2, 9, 2, 1, 9, 2, 1});
  // A legal merge embedded in a larger map exercises interactions with other
  // strides, not just injectivity of the merged pair itself.
  for (int64_t inner = -2; inner <= 2; ++inner)
    for (int64_t other = -5; other <= 5; ++other)
      CheckAddressFusion({1, 64, 64, 1, 3, 24, 24, 2, 2, 2,
                          2 * inner, inner, other, 2 * inner, inner, other});
  for (int64_t other : {4, 6}) {
    const std::vector<int64_t> words{
        1, 32, 32, 1, 3, 0, 0, 2, 2, 2, 6, 3, other, 6, 3, other};
    const auto record = descriptor::DecodeAddressLayout(words.data(), words.size()).records[0];
    assert(!InjectivityProof(record, false));
    assert(!InjectivityProof(record, true));
    const auto addresses = Enumerate(record);
    bool unique = true;
    for (std::size_t i = 0; i < addresses.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (addresses[i].second == addresses[j].second) unique = false;
    // Stride 4 is actually injective but fails the sufficient proof; stride 6
    // has genuine collisions. Both must retain their proof result after fusion.
    assert(unique == (other == 4));
    CheckAddressFusion(words);
    // Add a zero-destination fiber: Classify must apply the same proof to just
    // the map subset, whereas the full-axis proof also sees the zero stride.
    CheckAddressFusion({1, 32, 32, 1, 4, 0, 0, 2, 2, 2, 2,
                        6, 3, other, 0, 6, 3, other, 0});
  }
  CheckAddressFusion({1, 8, 8, 1, 3, 0, 0, 2, 2, 2,
                      2, 1, 0, 2, 1, 0});
  // Alias identity ignores singleton strides, but compares every empty stride.
  for (const auto& words : {
      std::vector<int64_t>{1, 6, 6, 1, 3, 0, 0, 1, 2, 3,
                           INT64_MIN, 3, 1, INT64_MAX, 3, 1},
      std::vector<int64_t>{1, 6, 6, 1, 3, 0, 0, 0, 2, 3,
                           INT64_MIN, 3, 1, INT64_MAX, 3, 1}}) {
    const auto record = descriptor::DecodeAddressLayout(words.data(), words.size()).records[0];
    assert(AliasProof(record) == (record.shape[0] == 1));
    CheckAddressFusion(words);
  }
  // The count fits u64 but a fused ordinary extent would not fit signed V1.
  CheckAddressFusion({1, 1, 1, 1, 2, 0, 0, INT64_MAX, 2, 0, 0, 0, 0}, false);
  CheckAddressFusion({1, 1, 1, 1, 2, 0, 0, INT64_MAX / 2, 2, 0, 0, 0, 0}, false);
}

void CheckReductionFusion(const std::vector<uint64_t>& words,
                          bool enumerate = true) {
  const auto bytes = Bytes(words);
  const auto before = descriptor::DecodeReductionLayout(bytes.data(), bytes.size());
  std::vector<int64_t> signed_words;
  for (auto word : words) signed_words.push_back(static_cast<int64_t>(word));
  const auto prepared = PrepareLayout(4, signed_words);
  assert(prepared.status == 0);
  assert(PrepareLayout(4, prepared.words).words == prepared.words);
  const auto output = Bytes(std::vector<uint64_t>(prepared.words.begin(), prepared.words.end()));
  const auto after = descriptor::DecodeReductionLayout(output.data(), output.size());
  assert(before.source_size == after.source_size && before.output_size == after.output_size);
  assert(after.records.size() == 1);
  const auto& a = before.records[0];
  const auto& b = after.records[0];
  assert(b.layout.semantic_index == a.layout.semantic_index);
  assert(b.layout.source_offset == a.layout.source_offset);
  assert(b.layout.destination_offset == a.layout.destination_offset);
  assert(layout::ElementCount(a.layout) == layout::ElementCount(b.layout));
  for (bool role : {false, true}) {
    uint64_t before_count = 1, after_count = 1;
    for (std::size_t i = 0; i < a.reduction_axes.size(); ++i)
      if (a.reduction_axes[i] == role) before_count *= a.layout.shape[i];
    for (std::size_t i = 0; i < b.reduction_axes.size(); ++i)
      if (b.reduction_axes[i] == role) after_count *= b.layout.shape[i];
    assert(before_count == after_count);
  }
  if (enumerate) {
    auto original = Enumerate(a.layout), normalized = Enumerate(b.layout);
    std::sort(original.begin(), original.end());
    std::sort(normalized.begin(), normalized.end());
    assert(original == normalized);
  }
}

void CheckReductionFusions() {
  CheckReductionFusion({1, 24, 6, 1, 4, 0, 0,
      2, 3, 2, 2, 12, 4, 2, 1, 2, 3, 1, 1, 3, 1, 99, 99, 0, 0, 1, 1});
  CheckReductionFusion({1, 1, 2, 1, 3, 0, 0,
      2, 2, 2, 0, 0, 0, 1, 2, 1, 0, 1, 0, 1, 0, 1});
  CheckReductionFusion({1, 1, 1, 1, 2, 0, 0,
      INT64_MAX, 2, 0, 0, 1, 1, 99, 99, 1, 1}, false);
  CheckReductionFusion({1, 1, 1, 1, 2, 0, 0,
      UINT64_MAX / 3, 3, 0, 0, 1, 1, 99, 99, 1, 1}, false);
  // Singleton reduction axes disappear; empty records retain their raw axes.
  CheckReductionFusion({1, 1, 1, 1, 3, 0, 0,
      2, 1, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1});
  CheckReductionFusion({1, 0, 1, 1, 3, 0, 0,
      0, 2, 3, 0, 3, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1});
  CheckReductionFusion({1, 1, 1, 1, 0, 0, 0});
  for (int64_t inner = -2; inner <= 2; ++inner) {
    CheckReductionFusion({1, 32, 1, 1, 2, 12, 0,
        2, 3, static_cast<uint64_t>(3 * inner), static_cast<uint64_t>(inner),
        1, 1, INT64_MAX, INT64_MAX, 1, 1});
  }
}

void CheckSortedPreparation() {
  // Column-major contiguity requires reordering before fusion. The canonical
  // descriptor is stable even when already-prepared words enter CPU instantiate.
  const std::vector<int64_t> column{1, 6, 6, 1, 2, 0, 0, 2, 3, 1, 2, 1, 2};
  for (int operation = 0; operation < 4; ++operation) {
    const auto prepared = PrepareLayout(operation, column);
    assert(prepared.status == 0);
    assert((prepared.words == std::vector<int64_t>{1, 6, 6, 1, 1, 0, 0, 6, 1, 1}));
    assert(PrepareLayout(operation, prepared.words).words == prepared.words);
  }
  const std::vector<int64_t> singleton_rerank{
      1, 9, 15, 1, 3, 2, 0, 2, 1, 3, 4, 0, -1, 2, -1, 6};
  for (int operation : {0, 1}) {
    const auto prepared = PrepareLayout(operation, singleton_rerank);
    assert(prepared.status == 0);
    assert((prepared.words == std::vector<int64_t>{
        1, 9, 15, 1, 2, 2, 0, 2, 3, 4, -1, 2, 6}));
    assert(PrepareLayout(operation, prepared.words).words == prepared.words);
  }
  // Stable ties, negative and broadcast strides, nontrivial singleton removal,
  // and rank zero must not lead to a second normalization with different axes.
  std::mt19937_64 random(20260926);
  for (int iteration = 0; iteration < 15000; ++iteration) {
    const int rank = random() % 7;
    layout::Record address;
    layout::ReductionRecord reduction;
    for (int axis = 0; axis < rank; ++axis) {
      const auto extent = 1 + random() % 4;
      address.shape.push_back(extent);
      address.source_strides.push_back(static_cast<int64_t>(random() % 11) - 5);
      address.destination_strides.push_back(static_cast<int64_t>(random() % 11) - 5);
      reduction.reduction_axes.push_back(random() % 2);
    }
    reduction.layout = address;
    for (int axis = 0; axis < rank; ++axis)
      if (reduction.reduction_axes[axis]) reduction.layout.destination_strides[axis] = 0;
    auto once = address;
    layout::OptimizeRecordForPreparation(once);
    auto twice = once;
    layout::OptimizeRecordForPreparation(twice);
    SameRecord(once, twice);
    layout::OptimizeRecordForPreparation(reduction);
    auto roles_twice = reduction;
    layout::OptimizeRecordForPreparation(roles_twice);
    SameRecord(reduction.layout, roles_twice.layout);
    assert(reduction.reduction_axes == roles_twice.reduction_axes);
  }
  const std::vector<int64_t> ordered_roles{
      1, 24, 6, 1, 4, 0, 0,
      2, 3, 2, 2, 12, 4, 2, 1,
      2, 3, 1, 1, 3, 1, 99, 99, 0, 0, 1, 1};
  const auto role_result = PrepareLayout(4, ordered_roles);
  assert(role_result.status == 0);
  assert((role_result.words == std::vector<int64_t>{
      1, 24, 6, 1, 2, 0, 0, 4, 6, 1, 4, 1, 6, 0, 1, 1, 0}));
  // Unlike the map protocol, reduction fusion may produce unsigned extents.
  const std::vector<int64_t> unsigned_fiber{
      1, 1, 1, 1, 2, 0, 0, INT64_MAX, 2, 0, 0, 1, 1, 99, 99, 1, 1};
  const auto unsigned_result = PrepareLayout(4, unsigned_fiber);
  assert(unsigned_result.status == 0);
  assert((unsigned_result.words == std::vector<int64_t>{
      1, 1, 1, 1, 1, 0, 0, -2, 0, 1, 0, 1}));
  const std::vector<int64_t> empty_axes{
      1, 0, 0, 1, 2, 0, 0, 0, 2, INT64_MIN, 1, INT64_MAX, 1};
  assert(PrepareLayout(3, empty_axes).words == empty_axes);
  const layout::ReductionRecord singleton_map{{0, 0, 0, {1, 2}, {INT64_MIN, 1}, {0, 0}},
                                                {false, true}};
  auto prepared_map = singleton_map;
  layout::OptimizeRecordForPreparation(prepared_map);
  assert((prepared_map.layout.shape == std::vector<uint64_t>{2}));
  assert((prepared_map.reduction_axes == std::vector<bool>{true}));
}

void CheckOwnerFiberPacking() {
  const std::vector<int64_t> accumulation{
      1, 12, 8, 1, 4, 4, 1, 2, 3, 1, 2, 6, -2, INT64_MIN, 1, 1, 0, 0, 0};
  const auto decoded = descriptor::DecodeAccumulationLayout(accumulation.data(), accumulation.size());
  const auto packed = layout::PackOwnerFiber(decoded);
  assert(packed.schedules.size() == 1 && packed.schedules[0].owners == 2 &&
         packed.schedules[0].contributions == 6 && packed.schedules[0].cursor == 4);
  assert((packed.words == std::vector<int64_t>{1, 12, 8, 1,
      2, 2, 4, 1, 2, 1, 3, 2, 6, INT64_MIN, -2, 1, 1, 0, 0, 0}));
  const auto reduction_words = Bytes({1, 12, 8, 1, 4, 4, 1,
      2, 3, 1, 2, 6, static_cast<uint64_t>(-2LL), static_cast<uint64_t>(INT64_MIN), 1,
      2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1});
  const auto reduction = descriptor::DecodeReductionLayout(reduction_words.data(), reduction_words.size());
  const auto grouped = layout::PackOwnerFiber(reduction);
  assert(grouped.schedules.size() == 1 && grouped.schedules[0].owners == 2 &&
         grouped.schedules[0].contributions == 6 && grouped.words == packed.words);
  const auto collision = std::vector<int64_t>{1, 4, 3, 1, 2, 0, 0, 2, 2, 2, 1, 1, 1};
  ExpectInvalid([&] { descriptor::DecodeAccumulationLayout(collision.data(), collision.size()); });
  assert(descriptor::DecodeAddressLayout(collision.data(), collision.size()).records.size() == 1);
  // One common packed schema: map rank, fiber rank, two offsets, grouped
  // extents, first-lane strides, second-lane strides. Dot's second lane is
  // the right input; for other operations it is the destination.
  const std::vector<int64_t> map_words{1, 12, 8, 1, 2, 4, 1, 2, 3, 3, -1, 3, 1};
  const auto map = descriptor::DecodeLayout(map_words.data(), map_words.size());
  const auto dot = descriptor::DecodeAddressLayout(map_words.data(), map_words.size());
  const std::vector<int64_t> map_expected{1, 12, 8, 1,
      2, 0, 4, 1, 2, 3, 3, -1, 3, 1};
  const std::vector<int64_t> dot_expected{1, 12, 8, 1,
      0, 2, 4, 1, 2, 3, 3, -1, 3, 1};
  for (int operation : {0, 1}) {
    const auto result = layout::PackOwnerFiber(map, operation);
    assert(result.words == map_expected && result.schedules.size() == 1 &&
           result.schedules[0].owners == 6 && result.schedules[0].contributions == 1 &&
           result.schedules[0].cursor == 4);
  }
  const auto dot_result = layout::PackOwnerFiber(dot, 2);
  assert(dot_result.words == dot_expected && dot_result.schedules.size() == 1 &&
         dot_result.schedules[0].owners == 1 && dot_result.schedules[0].contributions == 6);
  for (const auto& [operation, words, expected] : {
      std::tuple<int32_t, std::vector<int64_t>, std::vector<int64_t>>{0, map_words, map_expected},
      {1, map_words, map_expected}, {2, map_words, dot_expected},
      {3, accumulation, packed.words},
      {4, std::vector<int64_t>{1, 12, 8, 1, 4, 4, 1,
          2, 3, 1, 2, 6, -2, INT64_MIN, 1, 2, 1, 1, 1,
          1, 0, 0, 0, 0, 1, 0, 1}, grouped.words}}) {
    BridgeResult result;
    Tensor0StridePackOwnerFiber(operation, words.data(), words.size(), &result, ReceiveLayout);
    assert(result.calls == 1 && result.status == 0 && result.words == expected);
  }
  // Reduction destination strides on reduction axes are ignored by the V1
  // protocol, not inferred as map roles from their incoming values.
  auto noncanonical = std::vector<int64_t>{1, 12, 8, 1, 4, 4, 1,
      2, 3, 1, 2, 6, -2, INT64_MIN, 1, 2, 1, 1, 1,
      1, 0, 0, 0, 0, 1, 0, 1};
  noncanonical[20] = 99;
  noncanonical[22] = -99;
  BridgeResult ignored;
  Tensor0StridePackOwnerFiber(4, noncanonical.data(), noncanonical.size(), &ignored, ReceiveLayout);
  assert(ignored.calls == 1 && ignored.status == 0 && ignored.words == grouped.words);
  BridgeResult rejected;
  Tensor0StridePackOwnerFiber(3, collision.data(), collision.size(), &rejected, ReceiveLayout);
  assert(rejected.calls == 1 && rejected.status == 1 &&
         rejected.error.find("cannot prove injective output owners") != std::string::npos);
}

int main() {
  CheckOwnerFiberPacking();
  CheckLayout();
  CheckReduction();
  CheckRoundTrips();
  CheckSortedPreparation();
  CheckPreparationBridge();
  CheckFusion();
  CheckReductionFusions();
}
