#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "layout/blocking.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "execute/scheduling.inc"
#include "execute/map.inc"

template <typename Function>
void VisitAddresses(const layout::Record& record, Function function) {
  for (uint64_t linear = 0; linear < layout::ElementCount(record); ++linear) {
    auto remaining = linear;
    auto source = record.source_offset;
    auto destination = record.destination_offset;
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      const auto coordinate = static_cast<int64_t>(remaining % record.shape[axis]);
      remaining /= record.shape[axis];
      source += coordinate * record.source_strides[axis];
      destination += coordinate * record.destination_strides[axis];
    }
    function(source, destination);
  }
}

void CheckBlocks(const layout::Record& record, bool require_multiple = false) {
  const auto program = layout::CompileGeneratedRecord(record, true, true, 4, 4);
  std::map<std::pair<int64_t, int64_t>, int> expected;
  VisitAddresses(record, [&](auto source, auto destination) { ++expected[{source, destination}]; });
  decltype(expected) actual;
  std::size_t count = 0;
  layout::ForEachGeneratedBlock(program, [&](const auto& block) {
    ++count;
    assert(block.semantic_index == record.semantic_index);
    VisitAddresses(block, [&](auto source, auto destination) { ++actual[{source, destination}]; });
  });
  assert(actual == expected);
  if (require_multiple) assert(count > 1);
}

void CheckSingleAxisPrograms() {
  layout::Record scalar;
  scalar.semantic_index = 7;
  scalar.source_offset = 3;
  scalar.destination_offset = 5;
  const auto scalar_program = layout::CompileGeneratedRecord(scalar, true, true, 4, 8);
  assert(scalar_program.blocks.empty() && scalar_program.split_costs.empty());
  assert(scalar_program.record.semantic_index == 7);
  assert(scalar_program.record.source_offset == 3 && scalar_program.record.destination_offset == 5);

  // Planning-only strides include saturation limits; no storage is dereferenced.
  const std::array<int64_t, 7> strides{INT64_MIN, -65537, -1, 0, 1, 65537, INT64_MAX};
  for (const auto source_stride : strides) {
    for (const auto destination_stride : strides) {
      for (const uint64_t extent : {UINT64_C(0), UINT64_C(1), UINT64_C(13), UINT64_C(200003)}) {
        const layout::Record record{11, 17, 23, {extent}, {source_stride}, {destination_stride}};
        const auto minimum = std::min(layout::AbsoluteStride(source_stride),
                                      layout::AbsoluteStride(destination_stride));
        const auto cost = minimum == 0 ? 1 :
            std::min<uint64_t>(minimum, UINT64_MAX / 2) * 2;
        for (const bool reorder : {false, true}) {
          for (const bool blocking : {false, true}) {
            const auto program = layout::CompileGeneratedRecord(record, reorder, blocking, 1, 16);
            assert(program.blocks == record.shape);
            assert(program.split_costs == std::vector<uint64_t>{cost});
            assert(program.record.shape == record.shape);
            assert(program.record.source_strides == record.source_strides);
            assert(program.record.destination_strides == record.destination_strides);
            assert(program.record.source_offset == 17 && program.record.destination_offset == 23);
            assert(program.record.semantic_index == 11);
          }
        }
      }
    }
  }
}

int main() {
  CheckSingleAxisPrograms();
  const auto line = layout::BuildLayout({200003}, {-2}, 400004, {3}, 1, 400005, 600010, 11);
  const auto line_program = layout::CompileGeneratedRecord(line, true, true, 4, 4);
  std::vector<layout::GeneratedRecordProgram> subdomains;
  layout::AppendGeneratedSubdomains(line_program, 3, &subdomains);
  assert(subdomains.size() == 3);
  uint64_t start = 0;
  for (std::size_t index = 0; index < subdomains.size(); ++index) {
    const auto& subdomain = subdomains[index];
    const uint64_t extent = index == 0 ? 100001 : 50001;
    assert(subdomain.record.shape == std::vector<uint64_t>{extent});
    assert(subdomain.record.source_offset == 400004 - static_cast<int64_t>(2 * start));
    assert(subdomain.record.destination_offset == 1 + static_cast<int64_t>(3 * start));
    assert(subdomain.record.semantic_index == 11);
    assert(subdomain.blocks[0] <= extent);
    start += extent;
  }
  assert(start == 200003);
  const auto record = layout::BuildLayout({129, 131}, {1, 129}, 0, {131, 1}, 2,
                                         129 * 131, 129 * 131 + 4, 7);
  CheckBlocks(record, true);
  auto reverse = record;
  reverse.source_offset = 128;
  reverse.source_strides[0] = -1;
  CheckBlocks(reverse, true);
  auto broadcast = record;
  broadcast.source_strides[0] = 0;
  CheckBlocks(broadcast);
  auto high_rank = record;
  high_rank.shape.resize(12, 1);
  high_rank.source_strides.resize(12, INT64_MAX);
  high_rank.destination_strides.resize(12, INT64_MAX);
  CheckBlocks(high_rank, true);
  CheckBlocks(layout::BuildLayout({}, {}, 0, {}, 2, 1, 3, 0));
  layout::GeneratedRecordProgram empty;
  empty.record.shape = {0};
  empty.record.source_strides = {1};
  empty.record.destination_strides = {1};
  empty.blocks = {1};
  layout::ForEachGeneratedBlock(empty, [](const auto&) { assert(false); });

  auto small = layout::BuildLayout({5, 3}, {1, 5}, 0, {3, 1}, 0, 15, 15, 9);
  layout::GeneratedRecordProgram manual{small, {2, 2}, {}};
  std::vector<int64_t> offsets;
  std::vector<std::vector<uint64_t>> shapes;
  layout::ForEachGeneratedBlock(manual, [&](const auto& block) {
    offsets.push_back(block.source_offset);
    shapes.push_back(block.shape);
  });
  assert((offsets == std::vector<int64_t>{0, 2, 4, 10, 12, 14}));
  assert((shapes == std::vector<std::vector<uint64_t>>{{2, 2}, {2, 2}, {1, 2},
                                                                   {2, 1}, {2, 1}, {1, 1}}));

  constexpr std::size_t source_size = 129 * 131;
  constexpr std::size_t output_size = source_size + 4;
  std::vector<float> source(source_size), base(output_size), result(output_size);
  for (std::size_t index = 0; index < source.size(); ++index) source[index] = index % 17;
  for (std::size_t index = 0; index < base.size(); ++index) base[index] = index % 13;
  const auto scalar_record = layout::BuildLayout({}, {}, 0, {}, 0, source_size, output_size, 8);
  const std::vector<layout::Record> records{reverse, scalar_record};
  std::vector<double> copied(output_size), expected_copy(output_size);
  {
    const auto programs = layout::PrepareGeneratedRecords(
        records, sizeof(scalar::Value<scalar::F32>), sizeof(scalar::Value<scalar::F64>));
    ExecuteCopyBatch<scalar::F32, scalar::F64>(programs, source.data(), copied.data(), output_size);
  }
  for (const auto& selected : records) {
    VisitAddresses(selected, [&](auto input, auto output) { expected_copy[output] = source[input]; });
  }
  assert(copied == expected_copy);
  for (const float alpha : {0.0f, 1.0f, 2.0f}) {
    for (const float beta : {0.0f, 1.0f, 3.0f}) {
      auto expected = base;
      for (const auto& selected : records) {
        VisitAddresses(selected, [&](auto input, auto output) {
          expected[output] = alpha * source[input] + beta * base[output];
        });
      }
      const auto* input = alpha == 0 ? nullptr : source.data();
      {
        const auto programs = layout::PrepareGeneratedRecords(
            records, sizeof(*input), sizeof(scalar::Value<scalar::F32>));
        ExecuteUpdateBatch<scalar::F32, scalar::F32, scalar::F32>(
            programs, input, base.data(), result.data(), output_size, alpha, beta,
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
      }
      assert(result == expected);
      auto inplace = base;
      {
        const auto programs = layout::PrepareGeneratedRecords(
            records, sizeof(*input), sizeof(scalar::Value<scalar::F32>));
        ExecuteUpdateBatch<scalar::F32, scalar::F32, scalar::F32>(
            programs, input, inplace.data(), inplace.data(), output_size, alpha, beta,
            expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
      }
      assert(inplace == expected);
    }
  }
  auto same_address = record;
  same_address.source_offset = same_address.destination_offset;
  same_address.source_strides = same_address.destination_strides;
  auto inplace = base;
  auto expected = base;
  VisitAddresses(same_address, [&](auto, auto output) { expected[output] *= 2; });
  {
    const auto programs = layout::PrepareGeneratedRecords(
        {same_address}, sizeof(*inplace.data()), sizeof(scalar::Value<scalar::F32>));
    ExecuteUpdateBatch<scalar::F32, scalar::F32, scalar::F32>(
        programs, inplace.data(), inplace.data(), inplace.data(), output_size, 2, 0,
        expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  }
  assert(inplace == expected);
}
