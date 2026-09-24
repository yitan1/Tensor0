#include "layout/descriptor.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
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
  const auto& record = decoded.records[0];
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
  for (const auto& invalid : {prefix_overflow, prefix_overflow + std::string(8, '\0')}) {
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
  assert((huge_decoded.records[0].shape == std::vector<uint64_t>{UINT64_MAX}));
  assert((huge_decoded.records[0].destination_strides == std::vector<int64_t>{0}));
  huge_words[5] = UINT64_C(1) << 63;
  const auto invalid_offset = Bytes(huge_words);
  ExpectInvalid([&] {
    descriptor::DecodeReductionLayout(invalid_offset.data(), invalid_offset.size());
  });
}

int main() {
  CheckLayout();
  CheckReduction();
}
