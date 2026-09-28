#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"

using namespace tensor0::stride;
#include <cassert>
#include <random>


void SameRecord(const layout::Record& actual, const layout::Record& expected) {
  assert(actual.semantic_index == expected.semantic_index);
  assert(actual.shape == expected.shape);
  assert(actual.source_offset == expected.source_offset);
  assert(actual.destination_offset == expected.destination_offset);
  assert(actual.source_strides == expected.source_strides);
  assert(actual.destination_strides == expected.destination_strides);
}

void CheckUnchangedFields() {
  const std::vector<uint64_t> shape{2, 1, 3};
  const std::vector<int64_t> left{-3, INT64_MAX, -1};
  const std::vector<int64_t> right{3, INT64_MIN, 1};
  const layout::Record expected{37, 5, 2, shape, left, right};
  SameRecord(layout::BuildLayout(shape, left, 5, right, 2, 6, 8, 37), expected);
  SameRecord(layout::BuildReductionLayout(shape, left, 5, shape, right, 2,
      {false, false, false}, 6, 8, 37), expected);
  const auto reduced = layout::BuildReductionLayout(shape, left, 5, {2, 1, 1}, right, 2,
      {false, false, true}, 6, 8, 37);
  SameRecord(reduced, {37, 5, 2, shape, left, {3, INT64_MIN, 0}});
  assert((shape == std::vector<uint64_t>{2, 1, 3}));
  assert((right == std::vector<int64_t>{3, INT64_MIN, 1}));
  SameRecord(layout::BuildLayout({}, {}, 2, {}, 3, 3, 4, 9), {9, 2, 3, {}, {}, {}});
  const auto empty = layout::BuildReductionLayout({0}, {INT64_MIN}, 0, {1}, {INT64_MAX}, 2,
      {true}, 0, 3, 7);
  SameRecord(empty, {7, 0, 2, {0}, {INT64_MIN}, {0}});
  bool rejected = false;
  try {
    layout::BuildReductionLayout({0}, {INT64_MIN}, 0, {1}, {INT64_MAX}, 2, {true}, 0, 2, 7);
  } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}

void CheckDotAddresses() {
  std::mt19937_64 random(4261);
  for (std::size_t trial = 0; trial < 3000; ++trial) {
    std::vector<uint64_t> shape(random() % 5);
    std::vector<int64_t> left(shape.size()), right(shape.size());
    uint64_t count = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      shape[axis] = random() % 4;
      count *= shape[axis];
      left[axis] = static_cast<int64_t>(random() % 9) - 4;
      right[axis] = static_cast<int64_t>(random() % 9) - 4;
    }
    const int64_t left_offset = random() % 16;
    const int64_t right_offset = random() % 16;
    const uint64_t left_size = random() % 32;
    const uint64_t right_size = random() % 32;
    bool expected = true;
    if (count == 0) {
      expected = static_cast<uint64_t>(left_offset) <= left_size &&
                 static_cast<uint64_t>(right_offset) <= right_size;
    }
    for (uint64_t element = 0; element < count; ++element) {
      auto remaining = element;
      auto left_address = left_offset;
      auto right_address = right_offset;
      for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        const auto coordinate = static_cast<int64_t>(remaining % shape[axis]);
        remaining /= shape[axis];
        left_address += coordinate * left[axis];
        right_address += coordinate * right[axis];
      }
      expected = expected && left_address >= 0 && right_address >= 0 &&
                 static_cast<uint64_t>(left_address) < left_size &&
                 static_cast<uint64_t>(right_address) < right_size;
    }
    bool accepted = true;
    try {
      const auto record = layout::BuildLayout(shape, left, left_offset, right, right_offset,
          left_size, right_size, trial);
      SameRecord(record, {trial, left_offset, right_offset, shape, left, right});
    } catch (const std::invalid_argument&) { accepted = false; }
    assert(accepted == expected);
  }
}

int main() {
  CheckUnchangedFields();
  CheckDotAddresses();
}
