namespace layout {

struct GeneratedRecordProgram {
  Record record;
  std::vector<uint64_t> blocks;
  std::vector<uint64_t> split_costs;
};

uint64_t GeneratedMemoryRegion(
    const std::vector<uint64_t>& dims,
    const std::vector<std::vector<uint64_t>>& byte_strides) {
  constexpr uint64_t kCacheLineBytes = 64;
  uint64_t memory_region = 0;
  for (const auto& strides : byte_strides) {
    uint64_t contiguous_bytes = 0;
    uint64_t cache_line_blocks = 1;
    for (std::size_t axis = 0; axis < dims.size(); ++axis) {
      if (strides[axis] < kCacheLineBytes) {
        uint64_t axis_bytes = 0;
        if (!CheckedMultiply(dims[axis] - 1, strides[axis], &axis_bytes) ||
            !CheckedAdd(contiguous_bytes, axis_bytes,
                        &contiguous_bytes)) {
          return std::numeric_limits<uint64_t>::max();
        }
      } else if (!CheckedMultiply(cache_line_blocks, dims[axis],
                                  &cache_line_blocks)) {
        return std::numeric_limits<uint64_t>::max();
      }
    }
    uint64_t contiguous_lines = contiguous_bytes / kCacheLineBytes + 1;
    uint64_t operand_bytes = 0;
    if (!CheckedMultiply(kCacheLineBytes, contiguous_lines,
                         &operand_bytes) ||
        !CheckedMultiply(operand_bytes, cache_line_blocks,
                         &operand_bytes) ||
        !CheckedAdd(memory_region, operand_bytes, &memory_region)) {
      return std::numeric_limits<uint64_t>::max();
    }
  }
  return memory_region;
}

std::vector<uint64_t> ComputeGeneratedBlocks(
    const std::vector<uint64_t>& dims,
    const std::vector<uint64_t>& costs,
    const std::vector<std::vector<uint64_t>>& byte_strides,
    const std::vector<std::vector<uint64_t>>& stride_orders) {
  constexpr uint64_t kBlockMemoryBytes = UINT64_C(1) << 15;
  if (dims.empty() ||
      GeneratedMemoryRegion(dims, byte_strides) <= kBlockMemoryBytes) {
    return dims;
  }

  uint64_t minimum_order = std::numeric_limits<uint64_t>::max();
  for (const auto& orders : stride_orders) {
    minimum_order = std::min(
        minimum_order, *std::min_element(orders.begin(), orders.end()));
  }
  bool common_fastest_axis = true;
  for (const auto& orders : stride_orders) {
    common_fastest_axis =
        common_fastest_axis && orders.front() == minimum_order;
  }
  if (common_fastest_axis) {
    std::vector<uint64_t> tail_dims(dims.begin() + 1, dims.end());
    std::vector<uint64_t> tail_costs(costs.begin() + 1, costs.end());
    std::vector<std::vector<uint64_t>> tail_byte_strides;
    std::vector<std::vector<uint64_t>> tail_stride_orders;
    tail_byte_strides.reserve(byte_strides.size());
    tail_stride_orders.reserve(stride_orders.size());
    for (const auto& strides : byte_strides) {
      tail_byte_strides.emplace_back(strides.begin() + 1, strides.end());
    }
    for (const auto& orders : stride_orders) {
      tail_stride_orders.emplace_back(orders.begin() + 1, orders.end());
    }
    std::vector<uint64_t> result{dims.front()};
    const auto tail = ComputeGeneratedBlocks(
        tail_dims, tail_costs, tail_byte_strides, tail_stride_orders);
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
  }

  uint64_t minimum_byte_stride = std::numeric_limits<uint64_t>::max();
  for (const auto& strides : byte_strides) {
    minimum_byte_stride = std::min(
        minimum_byte_stride,
        *std::min_element(strides.begin(), strides.end()));
  }
  if (minimum_byte_stride > kBlockMemoryBytes) {
    return std::vector<uint64_t>(dims.size(), 1);
  }

  std::vector<uint64_t> blocks = dims;
  const auto split_axis = [&]() {
    std::size_t selected = 0;
    uint64_t selected_score = 0;
    for (std::size_t axis = 0; axis < blocks.size(); ++axis) {
      uint64_t score = 0;
      if (!CheckedMultiply(blocks[axis] - 1, costs[axis], &score)) {
        score = std::numeric_limits<uint64_t>::max();
      }
      if (score >= selected_score) {
        selected = axis;
        selected_score = score;
      }
    }
    return selected;
  };
  while (GeneratedMemoryRegion(blocks, byte_strides) >=
         2 * kBlockMemoryBytes) {
    const std::size_t axis = split_axis();
    blocks[axis] = (blocks[axis] + 1) >> 1;
  }
  while (GeneratedMemoryRegion(blocks, byte_strides) >
         kBlockMemoryBytes) {
    const std::size_t axis = split_axis();
    if (blocks[axis] <= 1) {
      break;
    }
    --blocks[axis];
  }
  return blocks;
}

GeneratedRecordProgram CompileGeneratedRecord(const Record& source,
                                               bool reorder_axes,
                                               bool blocking,
                                               uint64_t source_item_size,
                                               uint64_t destination_item_size) {
  GeneratedRecordProgram program;
  program.record = source;
  const std::size_t rank = source.shape.size();
  // With no competing axis, the common-fastest-axis rule keeps the whole domain.
  if (rank <= 1) {
    program.blocks = source.shape;
    if (rank == 1) {
      const auto stride = std::min(AbsoluteStride(source.source_strides[0]),
                                   AbsoluteStride(source.destination_strides[0]));
      program.split_costs = {stride == 0 ? 1 :
          std::min<uint64_t>(stride, std::numeric_limits<uint64_t>::max() / 2) * 2};
    }
    return program;
  }
  std::vector<int64_t> destination_strides(
      source.destination_strides.begin(),
      source.destination_strides.begin() + rank);
  std::vector<int64_t> source_strides(
      source.source_strides.begin(), source.source_strides.begin() + rank);
  const auto destination_order = GeneratedIndexOrder(destination_strides);
  const auto source_order = GeneratedIndexOrder(source_strides);
  std::vector<std::size_t> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);
  if (reorder_axes) {
    axes = ComputeLocalityOrder(source);
  }

  std::vector<uint64_t> dims(rank);
  std::vector<uint64_t> costs(rank);
  std::vector<std::vector<uint64_t>> byte_strides(2,
                                                   std::vector<uint64_t>(rank));
  std::vector<std::vector<uint64_t>> stride_orders(2,
                                                   std::vector<uint64_t>(rank));
  const auto byte_stride = [](uint64_t stride, uint64_t item_size) {
    uint64_t bytes = 0;
    return CheckedMultiply(stride, item_size, &bytes)
               ? bytes
               : std::numeric_limits<uint64_t>::max();
  };
  for (std::size_t position = 0; position < rank; ++position) {
    const std::size_t axis = axes[position];
    program.record.shape[position] = source.shape[axis];
    program.record.source_strides[position] = source.source_strides[axis];
    program.record.destination_strides[position] =
        source.destination_strides[axis];
    dims[position] = source.shape[axis];
    const uint64_t source_stride = AbsoluteStride(source.source_strides[axis]);
    const uint64_t destination_stride =
        AbsoluteStride(source.destination_strides[axis]);
    const uint64_t minimum_stride =
        std::min(source_stride, destination_stride);
    costs[position] =
        minimum_stride == 0
            ? 1
            : std::min<uint64_t>(minimum_stride,
                                 std::numeric_limits<uint64_t>::max() / 2) *
                  2;
    byte_strides[0][position] =
        byte_stride(destination_stride, destination_item_size);
    byte_strides[1][position] =
        byte_stride(source_stride, source_item_size);
    stride_orders[0][position] = destination_order[axis];
    stride_orders[1][position] = source_order[axis];
  }
  program.blocks =
      blocking
          ? ComputeGeneratedBlocks(dims, costs, byte_strides, stride_orders)
          : dims;
  program.split_costs = std::move(costs);
  return program;
}

std::vector<GeneratedRecordProgram> PrepareGeneratedRecords(
    const std::vector<Record>& records, uint64_t source_item_size, uint64_t destination_item_size,
    bool reorder_axes = true) {
  std::vector<GeneratedRecordProgram> programs;
  for (const auto& record : records) {
    if (ElementCount(record) == 0) continue;
    programs.push_back(CompileGeneratedRecord(record, reorder_axes, true, source_item_size, destination_item_size));
  }
  return programs;
}

void AppendGeneratedSubdomains(const GeneratedRecordProgram& program,
                               uint64_t thread_budget,
                               std::vector<GeneratedRecordProgram>* output) {
  constexpr uint64_t kMinimumThreadElements = UINT64_C(1) << 15;
  const auto& record = program.record;
  if (thread_budget <= 1 || ElementCount(record) <= kMinimumThreadElements || record.shape.empty()) {
    output->push_back(program);
    return;
  }
  std::size_t split_axis = 0;
  uint64_t split_score = 0;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    uint64_t score = 0;
    if (!CheckedMultiply(record.shape[axis] - 1, program.split_costs[axis], &score)) {
      score = std::numeric_limits<uint64_t>::max();
    }
    if (score >= split_score) {
      split_axis = axis;
      split_score = score;
    }
  }
  const auto dimension = record.shape[split_axis];
  if (program.split_costs[split_axis] == 0 ||
      dimension <= std::min<uint64_t>(program.blocks[split_axis], 1024)) {
    output->push_back(program);
    return;
  }
  const auto left_extent = dimension >> 1;
  if (left_extent == 0) {
    output->push_back(program);
    return;
  }
  auto left = program;
  auto right = program;
  left.record = SliceRecordRange(record, split_axis, 0, left_extent);
  right.record = SliceRecordRange(record, split_axis, left_extent, dimension - left_extent);
  left.blocks[split_axis] = std::min(left.blocks[split_axis], left_extent);
  right.blocks[split_axis] = std::min(right.blocks[split_axis], dimension - left_extent);
  const auto left_threads = thread_budget >> 1;
  AppendGeneratedSubdomains(left, left_threads, output);
  AppendGeneratedSubdomains(right, thread_budget - left_threads, output);
}

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

std::vector<std::vector<GeneratedRecordProgram>> BuildReductionOutputTasks(
    const std::vector<GeneratedRecordProgram>& programs, uint64_t desired_tasks) {
  struct Candidate {
    const GeneratedRecordProgram* program;
    std::vector<std::pair<int64_t, uint64_t>> output_axes;
    int64_t minimum;
    int64_t maximum;
  };
  std::vector<Candidate> candidates;
  for (const auto& program : programs) {
    const auto& record = program.record;
    if (ElementCount(record) == 0) continue;
    Candidate candidate{&program, {}, 0, 0};
    std::vector<std::size_t> axes;
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      if (record.shape[axis] > 1 && record.destination_strides[axis] != 0) {
        axes.push_back(axis);
        candidate.output_axes.emplace_back(record.destination_strides[axis], record.shape[axis]);
      }
    }
    try {
      ValidateInjectiveView(record, false, axes);
    } catch (const std::invalid_argument&) {
      return {};
    }
    AddressBounds(record, false, axes, &candidate.minimum, &candidate.maximum);
    std::sort(candidate.output_axes.begin(), candidate.output_axes.end());
    candidates.push_back(std::move(candidate));
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
    if (left.program->record.destination_offset != right.program->record.destination_offset) {
      return left.program->record.destination_offset < right.program->record.destination_offset;
    }
    return left.output_axes < right.output_axes;
  });
  std::vector<std::vector<GeneratedRecordProgram>> tasks;
  std::vector<std::pair<int64_t, int64_t>> intervals;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    if (index == 0 ||
        candidate.program->record.destination_offset != candidates[index - 1].program->record.destination_offset ||
        candidate.output_axes != candidates[index - 1].output_axes) {
      tasks.emplace_back();
      intervals.emplace_back(candidate.minimum, candidate.maximum);
    }
    tasks.back().push_back(*candidate.program);
  }
  std::sort(intervals.begin(), intervals.end());
  for (std::size_t index = 1; index < intervals.size(); ++index) {
    if (intervals[index - 1].second >= intervals[index].first) return {};
  }
  while (tasks.size() < desired_tasks) {
    std::size_t split_task = tasks.size();
    std::size_t split_axis = 0;
    uint64_t best_cost = 0;
    for (std::size_t index = 0; index < tasks.size(); ++index) {
      const auto& record = tasks[index].front().record;
      for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
        if (record.shape[axis] <= 1 || record.destination_strides[axis] == 0) continue;
        uint64_t cost = 0;
        if (!CheckedMultiply(record.shape[axis] - 1, AbsoluteStride(record.destination_strides[axis]), &cost)) {
          cost = std::numeric_limits<uint64_t>::max();
        }
        if (cost >= best_cost) {
          best_cost = cost;
          split_task = index;
          split_axis = axis;
        }
      }
    }
    if (split_task == tasks.size()) break;
    const auto extent = tasks[split_task].front().record.shape[split_axis];
    const auto stride = tasks[split_task].front().record.destination_strides[split_axis];
    const auto left_extent = extent / 2;
    std::vector<GeneratedRecordProgram> right;
    for (auto& program : tasks[split_task]) {
      auto& record = program.record;
      for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
        if (record.shape[axis] != extent || record.destination_strides[axis] != stride) continue;
        right.push_back(program);
        right.back().record = SliceRecordRange(record, axis, left_extent, extent - left_extent);
        right.back().blocks[axis] = std::min(program.blocks[axis], extent - left_extent);
        record = SliceRecordRange(record, axis, 0, left_extent);
        program.blocks[axis] = std::min(program.blocks[axis], left_extent);
        break;
      }
    }
    tasks.insert(tasks.begin() + static_cast<std::ptrdiff_t>(split_task + 1), std::move(right));
  }
  return tasks;
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
