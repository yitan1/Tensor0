#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/planning.inc"
#include "layout/blocking.inc"

#include <cassert>
#include <random>

template <typename Record>
constexpr bool kHasOnlyLayoutFields =
    !requires(Record record) { record.rank; } &&
    !requires(Record record) { record.logical_elements; } &&
    !requires(Record record) { record.source_size; } &&
    !requires(Record record) { record.output_size; } &&
    !requires(Record record) { record.scale; } &&
    !requires(Record record) { record.scale_real_bits; } &&
    !requires(Record record) { record.scale_imaginary_bits; } &&
    !requires(Record record) { record.mapped_dtype; } &&
    !requires(Record record) { record.identity_scale; };

static_assert(kHasOnlyLayoutFields<layout::Record>);

using Addresses = std::vector<std::pair<int64_t, int64_t>>;

template <typename Record>
void Enumerate(const Record& record, std::size_t axis, int64_t source,
               int64_t destination, Addresses* addresses) {
  if (axis == record.shape.size()) {
    addresses->emplace_back(source, destination);
    return;
  }
  for (uint64_t index = 0; index < record.shape[axis]; ++index) {
    Enumerate(record, axis + 1,
              source + static_cast<int64_t>(index) * record.source_strides[axis],
              destination + static_cast<int64_t>(index) * record.destination_strides[axis],
              addresses);
  }
}

template <typename Record>
Addresses Enumerate(const Record& record) {
  Addresses addresses;
  Enumerate(record, 0, record.source_offset, record.destination_offset, &addresses);
  std::sort(addresses.begin(), addresses.end());
  assert(addresses.size() == layout::ElementCount(record));
  return addresses;
}

layout::Record MakeRecord(std::vector<uint64_t> shape,
                                  std::vector<int64_t> source_strides,
                                  std::vector<int64_t> destination_strides,
                                  std::size_t semantic_index = 17) {
  layout::Record record;
  record.semantic_index = semantic_index;
  record.shape = std::move(shape);
  record.source_strides = std::move(source_strides);
  record.destination_strides = std::move(destination_strides);
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if (record.shape[axis] > 0) {
      record.source_offset -= std::min<int64_t>(
          0, static_cast<int64_t>(record.shape[axis] - 1) * record.source_strides[axis]);
      record.destination_offset -= std::min<int64_t>(
          0, static_cast<int64_t>(record.shape[axis] - 1) * record.destination_strides[axis]);
    }
  }
  return record;
}

layout::Record WithSortedDimensions(layout::Record record) {
  layout::SortRecordDimensions(&record);
  return record;
}

template <typename Function>
void ExpectInvalid(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

template <typename Record>
std::vector<std::size_t> ActiveAxes(const Record& record, bool source) {
  std::vector<std::size_t> axes;
  if (layout::ElementCount(record) == 0) return axes;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    if (!source || record.source_strides[axis] != 0) axes.push_back(axis);
  }
  return axes;
}

template <typename Record>
void CheckBounds(const Record& record) {
  const auto addresses = Enumerate(record);
  for (bool source : {false, true}) {
    int64_t minimum = 0;
    int64_t maximum = 0;
    layout::AddressBounds(record, source, ActiveAxes(record, source), &minimum, &maximum);
    if (addresses.empty()) {
      assert(minimum == layout::Offset(record, source));
      assert(maximum == minimum);
      continue;
    }
    int64_t expected_minimum = std::numeric_limits<int64_t>::max();
    int64_t expected_maximum = 0;
    for (const auto& address : addresses) {
      const auto offset = source ? address.first : address.second;
      expected_minimum = std::min(expected_minimum, offset);
      expected_maximum = std::max(expected_maximum, offset);
    }
    assert(minimum == expected_minimum);
    assert(maximum == expected_maximum);
  }
}

void CheckOptimization(layout::Record record) {
  const auto original = Enumerate(record);
  const auto semantic_index = record.semantic_index;
  CheckBounds(record);
  layout::ValidateInjectiveView(record, true, ActiveAxes(record, true));
  layout::ValidateInjectiveView(record, false, ActiveAxes(record, false));
  layout::OptimizeRecordForExecution(&record);
  assert(record.semantic_index == semantic_index);
  assert(Enumerate(record) == original);
  layout::OptimizeRecordForExecution(&record);
  assert(record.semantic_index == semantic_index);
  assert(Enumerate(record) == original);
  CheckBounds(record);
  for (bool reorder : {false, true}) {
    for (bool blocking : {false, true}) {
      const auto program = layout::CompileGeneratedRecord(record, reorder, blocking, 4, 8);
      assert(program.blocks.size() == record.shape.size());
      assert(program.split_costs.size() == record.shape.size());
      assert(program.record.semantic_index == semantic_index);
      assert(Enumerate(program.record) == original);
    }
  }
}

void ReferenceSort(layout::Record* record) {
  std::vector<uint64_t> importance(record->shape.size(), 0);
  for (std::size_t axis = 0; axis < record->shape.size(); ++axis) {
    if (record->shape[axis] <= 1) continue;
    for (const bool destination : {true, false}) {
      const auto& strides = destination ? record->destination_strides : record->source_strides;
      const auto stride = layout::AbsoluteStride(strides[axis]);
      std::size_t order = 1;
      if (stride != 0) {
        for (const auto candidate : strides) {
          if (candidate != 0 && layout::AbsoluteStride(candidate) < stride) ++order;
        }
      }
      importance[axis] += (destination ? UINT64_C(2) : UINT64_C(1))
                          << (2 * (record->shape.size() - order));
    }
  }
  std::vector<std::size_t> axes(record->shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  std::stable_sort(axes.begin(), axes.end(), [&](auto left, auto right) {
    return importance[left] > importance[right];
  });
  const auto before = *record;
  for (std::size_t axis = 0; axis < record->shape.size(); ++axis) {
    record->shape[axis] = before.shape[axes[axis]];
    record->source_strides[axis] = before.source_strides[axes[axis]];
    record->destination_strides[axis] = before.destination_strides[axes[axis]];
  }
}

void CheckStridedStrategy(layout::Record record) {
  auto expected = record;
  ReferenceSort(&expected);
  for (std::size_t axis = expected.shape.size(); axis > 1; --axis) {
    const auto outer = axis - 1;
    const auto inner = axis - 2;
    const auto extent = static_cast<int64_t>(expected.shape[inner]);
    if (expected.source_strides[outer] == extent * expected.source_strides[inner] &&
        expected.destination_strides[outer] == extent * expected.destination_strides[inner]) {
      expected.shape[inner] *= expected.shape[outer];
      expected.shape[outer] = 1;
    }
  }
  ReferenceSort(&expected);
  while (!expected.shape.empty() && expected.shape.back() == 1) {
    expected.shape.pop_back();
    expected.source_strides.pop_back();
    expected.destination_strides.pop_back();
  }
  layout::OptimizeRecordForExecution(&record);
  assert(record.shape == expected.shape);
  assert(record.source_strides == expected.source_strides);
  assert(record.destination_strides == expected.destination_strides);
  assert(record.shape.size() == expected.shape.size());
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
  }
}

void CheckLocalityOrdering() {
  auto reordered = MakeRecord({2, 3, 4}, {1, 8, 2}, {1, 8, 2});
  CheckStridedStrategy(reordered);
  layout::OptimizeRecordForExecution(&reordered);
  assert(reordered.shape == std::vector<uint64_t>{24});
  assert(reordered.source_strides == std::vector<int64_t>{1});
  assert(reordered.destination_strides == std::vector<int64_t>{1});
  auto conflict = MakeRecord({64, 64, 64}, {4096, 1, 64}, {1, 4096, 64});
  assert(layout::ComputeLocalityOrder(conflict) == (std::vector<std::size_t>{0, 1, 2}));
  CheckStridedStrategy(conflict);
  CheckStridedStrategy(MakeRecord({2, 3, 4}, {-1, -8, -2}, {-1, -8, -2}));
  CheckStridedStrategy(MakeRecord({2, 3, 4}, {0, 0, 0}, {1, 8, 2}));
  CheckStridedStrategy(MakeRecord({1, 1, 1}, {0, 0, 0}, {1, 2, 3}));
  std::mt19937 random(30192);
  for (int sample = 0; sample < 1000; ++sample) {
    auto record = MakeRecord({2, 3, 4, 1, 2}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
    for (auto* strides : {&record.source_strides, &record.destination_strides}) {
      for (auto& stride : *strides) stride = static_cast<int64_t>(random() % 17) - 8;
    }
    CheckStridedStrategy(record);
  }
  auto high_rank = MakeRecord(std::vector<uint64_t>(40, 1),
                             std::vector<int64_t>(40, 1),
                             std::vector<int64_t>(40, 1));
  high_rank.shape[39] = 2;
  layout::OptimizeRecordForExecution(&high_rank);
  assert(high_rank.shape == std::vector<uint64_t>{2});
}

void CheckRandomLayouts() {
  std::mt19937 random(87341);
  for (int sample = 0; sample < 250; ++sample) {
    const std::size_t rank = 1 + random() % 5;
    std::vector<uint64_t> shape(rank);
    std::vector<int64_t> source_strides(rank);
    std::vector<int64_t> destination_strides(rank);
    std::vector<std::size_t> order(rank);
    std::iota(order.begin(), order.end(), 0);
    for (auto& extent : shape) extent = 1 + random() % 4;
    for (auto* strides : {&source_strides, &destination_strides}) {
      std::shuffle(order.begin(), order.end(), random);
      int64_t stride = 1;
      for (const auto axis : order) {
        (*strides)[axis] = random() % 2 ? stride : -stride;
        stride *= static_cast<int64_t>(shape[axis]);
      }
    }
    if (sample % 3 == 0) source_strides[random() % rank] = 0;
    CheckOptimization(MakeRecord(shape, source_strides, destination_strides,
                                 static_cast<std::size_t>(sample + 10)));
  }
}

void CheckHighRank() {
  std::vector<uint64_t> shape(12, 2);
  std::vector<int64_t> strides(12);
  int64_t stride = 1;
  for (auto& value : strides) {
    value = stride;
    stride *= 2;
  }
  auto reducible = MakeRecord(shape, strides, strides);
  assert(WithSortedDimensions(reducible).shape.size() == 12);
  assert(layout::IsCompactSameMapping(reducible));
  CheckOptimization(reducible);
  auto reverse = strides;
  std::reverse(reverse.begin(), reverse.end());
  auto irreducible = MakeRecord(shape, strides, reverse);
  CheckOptimization(irreducible);
  layout::OptimizeRecordForExecution(&irreducible);
  assert(irreducible.shape.size() == 12);
  assert(!layout::IsCompactSameMapping(irreducible));
  auto empty = MakeRecord(std::vector<uint64_t>(40, 0),
                          std::vector<int64_t>(40, 1),
                          std::vector<int64_t>(40, 1));
  layout::OptimizeRecordForExecution(&empty);
  assert(empty.shape.size() == 40);
  assert(empty.shape.size() == 40);
}

void CheckInvalidAddresses() {
  auto record = MakeRecord({2}, {-1}, {1});
  record.source_offset = 0;
  int64_t minimum = 0;
  int64_t maximum = 0;
  ExpectInvalid([&] { layout::AddressBounds(record, true, ActiveAxes(record, true), &minimum, &maximum); });
  record.source_offset = -1;
  ExpectInvalid([&] { layout::AddressBounds(record, true, ActiveAxes(record, true), &minimum, &maximum); });
  record.source_offset = std::numeric_limits<int64_t>::max();
  record.source_strides[0] = 1;
  ExpectInvalid([&] { layout::AddressBounds(record, true, ActiveAxes(record, true), &minimum, &maximum); });
  record.source_offset = 0;
  record.source_strides[0] = std::numeric_limits<int64_t>::min();
  ExpectInvalid([&] { layout::AddressBounds(record, true, ActiveAxes(record, true), &minimum, &maximum); });
  record = MakeRecord({2, 2}, {2, 1}, {1, 1});
  ExpectInvalid([&] { layout::ValidateInjectiveView(record, false, ActiveAxes(record, false)); });
  record = MakeRecord({2}, {0}, {0});
  layout::ValidateInjectiveView(record, true, ActiveAxes(record, true));
  ExpectInvalid([&] { layout::ValidateInjectiveView(record, false, ActiveAxes(record, false)); });
}

void EnumerateBlocks(const layout::GeneratedRecordProgram& program,
                     const layout::Record& record, std::size_t axis,
                     Addresses* addresses, std::size_t* count) {
  assert(record.semantic_index == program.record.semantic_index);
  if (axis == record.shape.size()) {
    const auto block = Enumerate(record);
    addresses->insert(addresses->end(), block.begin(), block.end());
    ++*count;
    return;
  }
  const uint64_t size = program.blocks[axis];
  assert(size > 0 && size <= record.shape[axis]);
  for (uint64_t start = 0; start < record.shape[axis]; start += size) {
    const auto slice = layout::SliceRecordRange(
        record, axis, start, std::min(size, record.shape[axis] - start));
    EnumerateBlocks(program, slice, axis + 1, addresses, count);
  }
}

void CheckBlocks() {
  const auto record = WithSortedDimensions(
      MakeRecord({96, 160}, {-1, 96}, {160, 1}, 43));
  for (bool reorder : {false, true}) {
    const auto program = layout::CompileGeneratedRecord(record, reorder, true, 4, 8);
    Addresses addresses;
    std::size_t count = 0;
    EnumerateBlocks(program, program.record, 0, &addresses, &count);
    std::sort(addresses.begin(), addresses.end());
    assert(addresses == Enumerate(record));
    assert(count > 1);
  }
  const auto empty = layout::SliceRecordRange(record, 0, record.shape[0], 0);
  assert(empty.semantic_index == record.semantic_index);
  assert(Enumerate(empty).empty());
  ExpectInvalid([&] { layout::SliceRecordRange(record, record.shape.size(), 0, 1); });
  ExpectInvalid([&] { layout::SliceRecordRange(record, 0, record.shape[0], 1); });
  std::vector<int64_t> strides(12);
  for (std::size_t axis = 0; axis < strides.size(); ++axis) {
    strides[axis] = INT64_C(1) << axis;
  }
  auto reverse = strides;
  std::reverse(reverse.begin(), reverse.end());
  auto high_rank = MakeRecord(std::vector<uint64_t>(12, 2), strides, reverse);
  layout::OptimizeRecordForExecution(&high_rank);
  assert(high_rank.shape.size() == 12);
  for (bool reorder : {false, true}) {
    const auto program = layout::CompileGeneratedRecord(high_rank, reorder, true, 4, 8);
    Addresses addresses;
    std::size_t count = 0;
    EnumerateBlocks(program, program.record, 0, &addresses, &count);
    std::sort(addresses.begin(), addresses.end());
    assert(addresses == Enumerate(high_rank));
    assert(count > 1);
  }
}

void CheckPlanReuse() {
  const auto first = WithSortedDimensions(MakeRecord({3}, {1}, {1}, 4));
  auto second = WithSortedDimensions(MakeRecord({3}, {1}, {1}, 9));
  second.source_offset = 3;
  second.destination_offset = 3;
  const std::vector<layout::Record> plan{first, second};
  const std::array<int, 6> source{1, 2, 3, 4, 5, 6};
  for (const int factor : {2, -3}) {
    std::array<int, 10> coefficients{};
    coefficients[4] = factor;
    coefficients[9] = factor + 1;
    std::array<int, 6> output{};
    for (const auto& record : plan) {
      for (uint64_t index = 0; index < layout::ElementCount(record); ++index) {
        const auto slice = layout::SliceRecordRange(record, 0, index, 1);
        const auto addresses = Enumerate(slice);
        assert(addresses.size() == 1);
        output[addresses[0].second] =
            source[addresses[0].first] * coefficients[slice.semantic_index];
      }
    }
    const std::array<int, 6> expected{
        factor, 2 * factor, 3 * factor,
        4 * (factor + 1), 5 * (factor + 1), 6 * (factor + 1)};
    assert(output == expected);
  }
}

void CheckLayoutPredicates() {
  const auto compact = WithSortedDimensions(MakeRecord({3, 4}, {4, 1}, {4, 1}));
  assert(layout::IsCompactSameMapping(compact));
  assert(layout::IsContiguousInnerRowsCandidate(compact));
  assert(layout::GenericRowCount(compact) == 3);
  assert(compact.shape[0] == 4);
  const auto broadcast = WithSortedDimensions(MakeRecord({3, 4}, {1, 0}, {4, 1}));
  assert(layout::IsBroadcastContiguousRowCandidate(broadcast));
  assert(!layout::IsCompactSameMapping(broadcast));
  const auto forward = MakeRecord({3, 4}, {1, 3}, {4, 1});
  assert(layout::IsRank2ForwardCandidate(forward));
  assert(!layout::IsRank2ReverseCandidate(forward));
  const auto reverse = MakeRecord({3, 4}, {4, 1}, {1, 3});
  assert(layout::IsRank2ReverseCandidate(reverse));
  const auto two_pair =
      MakeRecord({2, 3, 8, 16}, {128, 256, 1, 8}, {384, 128, 16, 1});
  std::array<std::size_t, 4> axes;
  assert(layout::IsRank4TwoPairCandidate(two_pair, &axes));
  assert(!layout::IsRank4TwoPairCandidate(compact, &axes));
}

void CheckElementCount() {
  layout::Record record;
  assert(layout::ElementCount(record) == 1);
  record.shape = {2, 3};
  assert(layout::ElementCount(record) == 6);
  record.shape[1] = 0;
  assert(layout::ElementCount(record) == 0);
  record.shape = {std::numeric_limits<uint64_t>::max(), 2};
  ExpectInvalid([&] { layout::ElementCount(record); });
  record = MakeRecord({2, 3}, {3, 1}, {3, 1});
  const auto slice = layout::SliceRecordRange(record, 1, 1, 2);
  assert(layout::ElementCount(slice) == 4);
  assert(layout::ElementCount(record) == 6);
  assert(layout::ElementCount(layout::SliceRecordRange(record, 1, 3, 0)) == 0);
}

int main() {
  CheckElementCount();
  CheckOptimization(MakeRecord({}, {}, {}));
  CheckOptimization(MakeRecord({0, 3}, {3, 1}, {3, 1}));
  CheckOptimization(MakeRecord({1, 1, 1}, {17, -3, 0}, {4, 2, 1}));
  CheckOptimization(MakeRecord({3, 4}, {-4, -1}, {4, 1}));
  CheckRandomLayouts();
  CheckLocalityOrdering();
  CheckHighRank();
  CheckInvalidAddresses();
  CheckBlocks();
  CheckPlanReuse();
  CheckLayoutPredicates();
}
