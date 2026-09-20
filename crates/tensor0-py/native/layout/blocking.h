#pragma once

#include "record.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace tensor0::stride::layout {

struct GeneratedRecordProgram {
  Record record;
  std::vector<uint64_t> blocks;
  std::vector<uint64_t> split_costs;
};

uint64_t GeneratedMemoryRegion(
    const std::vector<uint64_t>& dims,
    const std::vector<std::vector<uint64_t>>& byte_strides);

std::vector<uint64_t> ComputeGeneratedBlocks(
    const std::vector<uint64_t>& dims,
    const std::vector<uint64_t>& costs,
    const std::vector<std::vector<uint64_t>>& byte_strides,
    const std::vector<std::vector<uint64_t>>& stride_orders);

GeneratedRecordProgram CompileGeneratedRecord(const Record& source,
                                               bool reorder_axes,
                                               bool blocking,
                                               uint64_t source_item_size,
                                               uint64_t destination_item_size);

std::vector<GeneratedRecordProgram> PrepareGeneratedRecords(
    const std::vector<Record>& records, uint64_t source_item_size, uint64_t destination_item_size,
    bool reorder_axes = true);

void AppendGeneratedSubdomains(const GeneratedRecordProgram& program,
                               uint64_t thread_budget,
                               std::vector<GeneratedRecordProgram>* output);

std::vector<std::vector<GeneratedRecordProgram>> BuildReductionOutputTasks(
    const std::vector<GeneratedRecordProgram>& programs, uint64_t desired_tasks);

template <typename LocalityCost>
void SplitReductionDomains(std::vector<GeneratedRecordProgram>* domains, uint64_t desired_tasks,
                           LocalityCost locality_cost) {
  while (domains->size() < desired_tasks) {
    std::size_t split_domain = domains->size();
    std::size_t split_axis = 0;
    uint64_t best_cost = 0;
    for (std::size_t index = 0; index < domains->size(); ++index) {
      const auto& record = (*domains)[index].record;
      for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
        if (record.shape[axis] <= 1) continue;
        uint64_t cost = 0;
        if (!CheckedMultiply(record.shape[axis] - 1, locality_cost(record, axis), &cost)) {
          cost = std::numeric_limits<uint64_t>::max();
        }
        if (cost >= best_cost && cost != 0) {
          best_cost = cost;
          split_domain = index;
          split_axis = axis;
        }
      }
    }
    if (split_domain == domains->size()) return;
    auto right = (*domains)[split_domain];
    const auto& record = right.record;
    const auto left_extent = record.shape[split_axis] / 2;
    auto& left = (*domains)[split_domain];
    left.record = SliceRecordRange(record, split_axis, 0, left_extent);
    left.blocks[split_axis] = std::min(left.blocks[split_axis], left_extent);
    right.record = SliceRecordRange(record, split_axis, left_extent, record.shape[split_axis] - left_extent);
    right.blocks[split_axis] = std::min(right.blocks[split_axis], right.record.shape[split_axis]);
    domains->insert(domains->begin() + static_cast<std::ptrdiff_t>(split_domain + 1), std::move(right));
  }
}

template <typename Function>
void ForEachGeneratedBlock(const GeneratedRecordProgram& program, Function function) {
  const auto& record = program.record;
  if (ElementCount(record) == 0) return;
  if (program.blocks == record.shape) {
    function(record);
    return;
  }
  const auto rank = record.shape.size();
  std::vector<uint64_t> starts(rank);
  Record block = record;
  while (true) {
    block.source_offset = record.source_offset;
    block.destination_offset = record.destination_offset;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      block.source_offset += static_cast<int64_t>(starts[axis]) * record.source_strides[axis];
      block.destination_offset += static_cast<int64_t>(starts[axis]) * record.destination_strides[axis];
      block.shape[axis] = std::min(std::max<uint64_t>(1, program.blocks[axis]),
                                   record.shape[axis] - starts[axis]);
    }
    function(block);
    std::size_t axis = 0;
    for (; axis < rank; ++axis) {
      if (block.shape[axis] < record.shape[axis] - starts[axis]) {
        starts[axis] += block.shape[axis];
        break;
      }
      starts[axis] = 0;
    }
    if (axis == rank) return;
  }
}

}
