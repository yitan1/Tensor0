void DecodeReductionOutput(const ReductionRecord& record,
                           uint64_t output_linear, int64_t* input_base,
                           int64_t* output_address) {
  uint64_t remaining_output = output_linear;
  *input_base = record.input_offset;
  *output_address = record.output_offset;
  for (std::size_t position = 0; position < record.map_rank; ++position) {
    const std::size_t axis = record.map_loop_order[position];
    const uint64_t coordinate = remaining_output % record.shape[axis];
    remaining_output /= record.shape[axis];
    *input_base += static_cast<int64_t>(coordinate) *
                   record.input_strides[axis];
    *output_address += static_cast<int64_t>(coordinate) *
                       record.output_strides[axis];
  }
}

struct AffineMapReduceSubdomain {
  std::size_t record_index = 0;
  uint64_t owner_index = 0;
  uint64_t batch_index = 0;
  std::array<int64_t, 2> operand_offsets{};
  uint64_t logical_elements = 0;
  std::array<uint64_t, kMaximumRank> shape{};
};

std::size_t MapReduceLoopRank(const ReductionRecord& record) {
  return record.reduction_rank;
}

std::size_t MapReduceLoopAxis(const ReductionRecord& record,
                              std::size_t position) {
  return record.reduction_loop_order[position];
}

int64_t MapReduceOperandStride(const ReductionRecord& record,
                               std::size_t operand, std::size_t axis) {
  return operand == 0 ? record.input_strides[axis] : 0;
}

std::size_t MapReduceLoopRank(const Record& record) { return record.rank; }

std::size_t MapReduceLoopAxis(const Record& record,
                              std::size_t position) {
  return record.loop_axes_fastest_first[position];
}

int64_t MapReduceOperandStride(const Record& record, std::size_t operand,
                               std::size_t axis) {
  return operand == 0 ? record.source_strides[axis]
                      : record.destination_strides[axis];
}

uint64_t MapReduceAxisSplitCost(const ReductionRecord& record,
                                const AffineMapReduceSubdomain& domain,
                                std::size_t axis) {
  if (domain.shape[axis] <= 1) {
    return 0;
  }
  uint64_t cost = 0;
  if (!CheckedMultiply(
          domain.shape[axis] - 1,
          std::max<uint64_t>(1, AbsoluteStride(record.input_strides[axis])),
          &cost)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return cost;
}

uint64_t MapReduceAxisSplitCost(const Record& record,
                                const AffineMapReduceSubdomain& domain,
                                std::size_t axis) {
  if (domain.shape[axis] <= 1) {
    return 0;
  }
  const uint64_t first_stride = AbsoluteStride(record.source_strides[axis]);
  const uint64_t second_stride =
      AbsoluteStride(record.destination_strides[axis]);
  const uint64_t locality_cost = std::max<uint64_t>(
      1, std::min(first_stride == 0 ? 1 : first_stride,
                  second_stride == 0 ? 1 : second_stride));
  uint64_t cost = 0;
  if (!CheckedMultiply(domain.shape[axis] - 1, locality_cost, &cost)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return cost;
}

template <typename RecordType>
void SplitAffineMapReduceSubdomains(
    const std::vector<RecordType>& records, uint64_t desired_tasks,
    std::vector<AffineMapReduceSubdomain>* domains) {
  while (domains->size() < desired_tasks) {
    std::size_t split_domain = domains->size();
    std::size_t split_axis = 0;
    uint64_t best_cost = 0;
    for (std::size_t domain_index = 0; domain_index < domains->size();
         ++domain_index) {
      const AffineMapReduceSubdomain& domain = (*domains)[domain_index];
      const RecordType& record = records[domain.record_index];
      const std::size_t loop_rank = MapReduceLoopRank(record);
      for (std::size_t position = 0; position < loop_rank; ++position) {
        const std::size_t axis = MapReduceLoopAxis(record, position);
        const uint64_t cost =
            MapReduceAxisSplitCost(record, domain, axis);
        if (cost >= best_cost && cost != 0) {
          best_cost = cost;
          split_domain = domain_index;
          split_axis = axis;
        }
      }
    }
    if (split_domain == domains->size()) {
      break;
    }
    AffineMapReduceSubdomain left = (*domains)[split_domain];
    AffineMapReduceSubdomain right = left;
    const RecordType& record = records[left.record_index];
    const uint64_t original_extent = left.shape[split_axis];
    const uint64_t left_extent = original_extent / 2;
    const uint64_t right_extent = original_extent - left_extent;
    left.shape[split_axis] = left_extent;
    right.shape[split_axis] = right_extent;
    left.logical_elements =
        left.logical_elements / original_extent * left_extent;
    right.logical_elements -= left.logical_elements;
    for (std::size_t operand = 0; operand < left.operand_offsets.size();
         ++operand) {
      right.operand_offsets[operand] +=
          static_cast<int64_t>(left_extent) *
          MapReduceOperandStride(record, operand, split_axis);
    }
    (*domains)[split_domain] = left;
    domains->insert(
        domains->begin() + static_cast<std::ptrdiff_t>(split_domain + 1),
        right);
  }
}

template <typename RecordType, typename ResultElement, typename MapOp,
          typename AddOp>
ResultElement ExecuteAffineMapReduceSubdomain(
    const RecordType& record, const AffineMapReduceSubdomain& domain,
    MapOp map_op, AddOp add_op) {
  ResultElement accumulator{};
  if (domain.logical_elements == 0) {
    return accumulator;
  }
  std::array<uint64_t, kMaximumRank> coordinates{};
  auto addresses = domain.operand_offsets;
  const std::size_t loop_rank = MapReduceLoopRank(record);
  for (uint64_t index = 0; index < domain.logical_elements; ++index) {
    accumulator = add_op(accumulator, map_op(addresses[0], addresses[1]));
    if (index + 1 == domain.logical_elements) {
      break;
    }
    for (std::size_t position = 0; position < loop_rank; ++position) {
      const std::size_t axis = MapReduceLoopAxis(record, position);
      if (coordinates[axis] + 1 < domain.shape[axis]) {
        ++coordinates[axis];
        for (std::size_t operand = 0; operand < addresses.size(); ++operand) {
          addresses[operand] +=
              MapReduceOperandStride(record, operand, axis);
        }
        break;
      }
      for (std::size_t operand = 0; operand < addresses.size(); ++operand) {
        addresses[operand] -= static_cast<int64_t>(coordinates[axis]) *
                              MapReduceOperandStride(record, operand, axis);
      }
      coordinates[axis] = 0;
    }
  }
  return accumulator;
}

AffineMapReduceSubdomain MakeReductionSubdomain(
    const ReductionRecord& record, std::size_t record_index,
    uint64_t owner_index, uint64_t batch_index, int64_t input_base) {
  AffineMapReduceSubdomain domain;
  domain.record_index = record_index;
  domain.owner_index = owner_index;
  domain.batch_index = batch_index;
  domain.operand_offsets[0] = input_base;
  domain.logical_elements = record.reduction_count;
  for (std::size_t position = 0; position < record.reduction_rank;
       ++position) {
    const std::size_t axis = record.reduction_loop_order[position];
    domain.shape[axis] = record.shape[axis];
  }
  return domain;
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename AddOp>
ResultElement ExecuteReductionSubdomain(
    const ReductionRecord& record,
    const AffineMapReduceSubdomain& domain, const SourceElement* source,
    ScalarOp scalar_op, AddOp add_op) {
  const auto map_op = [source, scalar_op](int64_t input_address, int64_t) {
    return scalar_op(source[input_address]);
  };
  return ExecuteAffineMapReduceSubdomain<ReductionRecord, ResultElement>(
      record, domain, map_op, add_op);
}

enum class ReductionWriteMode : uint8_t {
  kAssignReducedValue,
  kAccumulateExisting,
};

template <ReductionWriteMode Mode, typename ResultElement, typename AddOp>
void WriteReducedValue(ResultElement* destination, ResultElement value,
                       AddOp add_op) {
  if constexpr (Mode == ReductionWriteMode::kAssignReducedValue) {
    *destination = value;
  } else {
    *destination = add_op(*destination, value);
  }
}

template <ReductionWriteMode Mode, typename SourceElement,
          typename ResultElement, typename ScalarOp, typename AddOp>
void ExecuteReductionOutputRange(const ReductionRecord& record,
                                 const SourceElement* source,
                                 ResultElement* result,
                                 uint64_t output_start,
                                 uint64_t output_count, ScalarOp scalar_op,
                                 AddOp add_op) {
  if (output_count == 0) {
    return;
  }
  // Interleave independent output fibers while preserving the fixed
  // reduction order within every lane.
  constexpr uint64_t kOutputBlock = 8;
  if (record.map_rank == 1 && record.reduction_rank == 1 &&
      output_count >= kOutputBlock) {
    const std::size_t map_axis = record.map_loop_order[0];
    const std::size_t reduction_axis = record.reduction_loop_order[0];
    const uint64_t block_stop =
        output_start + (output_count / kOutputBlock) * kOutputBlock;
    for (uint64_t block_start = output_start; block_start < block_stop;
         block_start += kOutputBlock) {
      std::array<ResultElement, kOutputBlock> accumulators{};
      std::array<int64_t, kOutputBlock> input_addresses{};
      std::array<int64_t, kOutputBlock> output_addresses{};
      for (uint64_t lane = 0; lane < kOutputBlock; ++lane) {
        const int64_t coordinate =
            static_cast<int64_t>(block_start + lane);
        input_addresses[lane] =
            record.input_offset + coordinate * record.input_strides[map_axis];
        output_addresses[lane] =
            record.output_offset +
            coordinate * record.output_strides[map_axis];
      }
      for (uint64_t reduction_linear = 0;
           reduction_linear < record.reduction_count; ++reduction_linear) {
        for (uint64_t lane = 0; lane < kOutputBlock; ++lane) {
          accumulators[lane] = add_op(
              accumulators[lane],
              scalar_op(source[input_addresses[lane]]));
          input_addresses[lane] += record.input_strides[reduction_axis];
        }
      }
      for (uint64_t lane = 0; lane < kOutputBlock; ++lane) {
        WriteReducedValue<Mode>(result + output_addresses[lane],
                                accumulators[lane], add_op);
      }
    }
    output_count -= block_stop - output_start;
    output_start = block_stop;
    if (output_count == 0) {
      return;
    }
  }
  std::array<uint64_t, kMaximumRank> map_coordinates{};
  uint64_t remaining_output = output_start;
  int64_t input_base = record.input_offset;
  int64_t output_address = record.output_offset;
  for (std::size_t position = 0; position < record.map_rank; ++position) {
    const std::size_t axis = record.map_loop_order[position];
    const uint64_t coordinate = remaining_output % record.shape[axis];
    remaining_output /= record.shape[axis];
    map_coordinates[axis] = coordinate;
    input_base += static_cast<int64_t>(coordinate) *
                  record.input_strides[axis];
    output_address += static_cast<int64_t>(coordinate) *
                      record.output_strides[axis];
  }
  const uint64_t output_stop = output_start + output_count;
  for (uint64_t output_linear = output_start;
       output_linear < output_stop; ++output_linear) {
    const auto domain =
        MakeReductionSubdomain(record, 0, 0, 0, input_base);
    const ResultElement accumulator =
        ExecuteReductionSubdomain<SourceElement, ResultElement>(
            record, domain, source, scalar_op, add_op);
    WriteReducedValue<Mode>(result + output_address, accumulator, add_op);
    if (output_linear + 1 == output_stop) {
      break;
    }
    for (std::size_t position = 0; position < record.map_rank; ++position) {
      const std::size_t axis = record.map_loop_order[position];
      if (map_coordinates[axis] + 1 < record.shape[axis]) {
        ++map_coordinates[axis];
        input_base += record.input_strides[axis];
        output_address += record.output_strides[axis];
        break;
      }
      input_base -= static_cast<int64_t>(map_coordinates[axis]) *
                    record.input_strides[axis];
      output_address -= static_cast<int64_t>(map_coordinates[axis]) *
                        record.output_strides[axis];
      map_coordinates[axis] = 0;
    }
  }
}

template <typename SourceElement, typename ResultElement, typename BindMapping,
          typename AddOp>
struct GeneratedReductionState {
  std::shared_ptr<const ParsedReductionPlan> execution;
  const SourceElement* source = nullptr;
  ResultElement* result = nullptr;
  uint64_t batch_count = 0;
  uint64_t worker_count = 0;
  bool grouped_output_owner = false;
  BindMapping bind_mapping;
  AddOp add_op;
};

struct ReductionFiberSchedule {
  uint64_t independent_fibers = 0;
  uint64_t chunks_per_fiber = 0;
  uint64_t total_tasks = 0;
  uint64_t worker_count = 0;
  bool complete_output = false;
  std::vector<AffineMapReduceSubdomain> domains;
  std::vector<int64_t> output_addresses;
};

template <typename SourceElement, typename ResultElement>
ReductionFiberSchedule SelectReductionFiberSchedule(
    const ParsedReductionPlan& plan, uint64_t available_workers,
    uint64_t batch_count) {
  ReductionFiberSchedule schedule;
  if constexpr (!((std::is_same_v<SourceElement, float> &&
                   std::is_same_v<ResultElement, float>) ||
                  (std::is_same_v<SourceElement, std::complex<float>> &&
                   std::is_same_v<ResultElement, std::complex<float>>))) {
    return schedule;
  }
  if (available_workers < 2 || plan.records.empty()) {
    return schedule;
  }
  const ReductionRecord& first_record = plan.records.front();
  const bool single_record = plan.records.size() == 1;
  const uint64_t output_count = first_record.output_count;
  uint64_t total_reduction_count = 0;
  for (const ReductionRecord& record : plan.records) {
    if (record.output_count == 0 || record.reduction_count == 0 ||
        (!single_record &&
         (record.output_count != 1 || output_count != 1 ||
          !SameReductionOutputMap(first_record, record))) ||
        !CheckedAdd(total_reduction_count, record.reduction_count,
                    &total_reduction_count)) {
      return schedule;
    }
  }
  if (total_reduction_count < 2) {
    return schedule;
  }

  const uint64_t configured =
      reduction_fiber_parallel_mode_for_tests.load(std::memory_order_relaxed);
  const bool forced = configured == kReductionFiberParallelForced;
  if (configured == kReductionFiberParallelDisabled ||
      (configured != kReductionFiberParallelUseProductionMode && !forced)) {
    return schedule;
  }
  if (!CheckedMultiply(batch_count, output_count,
                       &schedule.independent_fibers) ||
      schedule.independent_fibers == 0 ||
      schedule.independent_fibers >= available_workers) {
    schedule.independent_fibers = 0;
    return schedule;
  }

  if (!forced && schedule.independent_fibers != 1) {
    schedule.independent_fibers = 0;
    return schedule;
  }

  uint64_t total_work_elements = 0;
  uint64_t total_work_bytes = 0;
  if (!forced &&
      (!CheckedMultiply(schedule.independent_fibers,
                        total_reduction_count, &total_work_elements) ||
       !CheckedMultiply(total_work_elements, sizeof(SourceElement),
                        &total_work_bytes) ||
       total_work_bytes < plan.parallel_minimum_bytes)) {
    schedule.independent_fibers = 0;
    return schedule;
  }

  const uint64_t desired_chunks = std::min(
      kReductionFiberMaximumChunks,
      (available_workers + schedule.independent_fibers - 1) /
          schedule.independent_fibers);
  uint64_t maximum_chunks = total_reduction_count;
  if (!forced) {
    const uint64_t minimum_chunk_elements = std::max<uint64_t>(
        1, (kReductionFiberMinimumChunkBytes + sizeof(SourceElement) - 1) /
               sizeof(SourceElement));
    maximum_chunks = total_reduction_count / minimum_chunk_elements;
  }
  schedule.chunks_per_fiber =
      std::min(desired_chunks, maximum_chunks);
  if (schedule.chunks_per_fiber < 2) {
    return ReductionFiberSchedule{};
  }
  schedule.output_addresses.reserve(schedule.independent_fibers);
  schedule.domains.reserve(
      static_cast<std::size_t>(schedule.independent_fibers) *
      std::max<std::size_t>(schedule.chunks_per_fiber,
                            plan.records.size()));
  for (uint64_t batch = 0; batch < batch_count; ++batch) {
    for (uint64_t output = 0; output < output_count; ++output) {
      const uint64_t owner = schedule.output_addresses.size();
      std::vector<AffineMapReduceSubdomain> owner_domains;
      owner_domains.reserve(plan.records.size());
      int64_t owner_output_address = 0;
      for (std::size_t record_index = 0;
           record_index < plan.records.size(); ++record_index) {
        const ReductionRecord& record = plan.records[record_index];
        int64_t input_base = 0;
        int64_t output_address = 0;
        DecodeReductionOutput(record, output, &input_base, &output_address);
        if (record_index == 0) {
          owner_output_address = output_address;
        } else if (output_address != owner_output_address) {
          return ReductionFiberSchedule{};
        }
        input_base += static_cast<int64_t>(batch * plan.input_size);
        owner_domains.push_back(MakeReductionSubdomain(
            record, record_index, owner, batch, input_base));
      }
      const uint64_t desired_owner_tasks = std::max<uint64_t>(
          schedule.chunks_per_fiber, owner_domains.size());
      SplitAffineMapReduceSubdomains(plan.records, desired_owner_tasks,
                                     &owner_domains);
      schedule.domains.insert(schedule.domains.end(),
                              owner_domains.begin(), owner_domains.end());
      schedule.output_addresses.push_back(
          static_cast<int64_t>(batch * plan.output_size) +
          owner_output_address);
    }
  }
  schedule.total_tasks = schedule.domains.size();
  if (schedule.total_tasks < 2) {
    return ReductionFiberSchedule{};
  }
  schedule.worker_count =
      std::min(available_workers, schedule.total_tasks);
  schedule.complete_output = output_count == plan.output_size;
  return schedule;
}

template <typename Element>
struct alignas(64) PaddedReductionPartial {
  Element value{};
};

template <typename SourceElement, typename ResultElement, typename BindMapping,
          typename AddOp>
struct FiberPartialReductionState {
  FiberPartialReductionState(
      std::shared_ptr<const ParsedReductionPlan> execution_data,
      const SourceElement* source_data, ResultElement* result_data,
      uint64_t batches, ReductionFiberSchedule schedule_data,
      uint64_t workers, BindMapping mapping_binder, AddOp add_function)
      : execution(std::move(execution_data)),
        source(source_data),
        result(result_data),
        batch_count(batches),
        schedule(schedule_data),
        worker_count(workers),
        partials(schedule_data.total_tasks),
        remaining_workers(workers),
        bind_mapping(std::move(mapping_binder)),
        add_op(std::move(add_function)) {}

  std::shared_ptr<const ParsedReductionPlan> execution;
  const SourceElement* source = nullptr;
  ResultElement* result = nullptr;
  uint64_t batch_count = 0;
  ReductionFiberSchedule schedule;
  uint64_t worker_count = 0;
  std::vector<PaddedReductionPartial<ResultElement>> partials;
  std::atomic<uint64_t> remaining_workers;
  std::atomic<bool> failed{false};
  BindMapping bind_mapping;
  AddOp add_op;
};

template <typename State>
void ExecuteFiberPartialTasks(State* state, uint64_t worker_index) {
  const ParsedReductionPlan& plan = *state->execution;
  for (uint64_t task = worker_index; task < state->schedule.total_tasks;
       task += state->worker_count) {
    const AffineMapReduceSubdomain& domain = state->schedule.domains[task];
    const auto& record = plan.records[domain.record_index];
    state->bind_mapping(record, [&](auto scalar_op) {
      state->partials[task].value = ExecuteReductionSubdomain<
          std::remove_const_t<std::remove_pointer_t<decltype(state->source)>>,
          std::remove_pointer_t<decltype(state->result)>>(
          record, domain, state->source, scalar_op, state->add_op);
    });
  }
}

template <typename State>
void MergeFiberPartials(State* state) {
  using ResultElement = std::remove_pointer_t<decltype(state->result)>;
  uint64_t domain_index = 0;
  for (uint64_t owner = 0;
       owner < state->schedule.output_addresses.size(); ++owner) {
    ResultElement accumulator{};
    while (domain_index < state->schedule.domains.size() &&
           state->schedule.domains[domain_index].owner_index == owner) {
      accumulator = state->add_op(
          accumulator, state->partials[domain_index].value);
      ++domain_index;
    }
    state->result[state->schedule.output_addresses[owner]] = accumulator;
  }
}

template <typename State, typename ExecuteWorker, typename Finish>
ffi::Future ExecuteStaticMapReduceWorkers(
    std::shared_ptr<State> state, ffi::ThreadPool thread_pool,
    uint64_t worker_count, ExecuteWorker execute_worker, Finish finish) {
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule(
          [state, completion, scheduled, execute_worker, finish]() mutable {
            try {
              execute_worker(state.get(), scheduled);
              const bool last = state->remaining_workers.fetch_sub(
                                    1, std::memory_order_acq_rel) == 1;
              if (last &&
                  !state->failed.load(std::memory_order_acquire)) {
                finish(state.get());
              }
              completion.CountDown();
            } catch (const std::exception&) {
              state->failed.store(true, std::memory_order_release);
              state->remaining_workers.fetch_sub(1,
                                                  std::memory_order_acq_rel);
              completion.CountDown(ffi::Error::Internal(
                  "tensor0-stride: contained map-reduce worker exception"));
            } catch (...) {
              state->failed.store(true, std::memory_order_release);
              state->remaining_workers.fetch_sub(1,
                                                  std::memory_order_acq_rel);
              completion.CountDown(ffi::Error::Internal(
                  "tensor0-stride: contained unknown map-reduce worker "
                  "exception"));
            }
          });
    } catch (...) {
      state->failed.store(true, std::memory_order_release);
      state->remaining_workers.fetch_sub(worker_count - scheduled,
                                         std::memory_order_acq_rel);
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: failed to schedule map-reduce worker"));
      break;
    }
  }
  return future;
}

template <typename SourceElement, typename ResultElement, typename BindMapping,
          typename AddOp>
ffi::Future ExecuteFiberPartialReduction(
    std::shared_ptr<const ParsedReductionPlan> execution,
    const SourceElement* source, ResultElement* result,
    ffi::ThreadPool thread_pool, uint64_t batch_count,
    const ReductionFiberSchedule& schedule, BindMapping bind_mapping,
    AddOp add_op) {
  uint64_t scratch_bytes = 0;
  if (!CheckedMultiply(
          schedule.total_tasks,
          sizeof(PaddedReductionPartial<ResultElement>), &scratch_bytes) ||
      scratch_bytes > kReductionFiberMaximumScratchBytes) {
    return ErrorFuture(Invalid("reduction fiber scratch exceeds native cap"));
  }
  const uint64_t worker_count = schedule.worker_count;
  using State = FiberPartialReductionState<SourceElement, ResultElement,
                                           BindMapping, AddOp>;
  auto state = std::make_shared<State>(
      std::move(execution), source, result, batch_count, schedule,
      worker_count, std::move(bind_mapping), std::move(add_op));
  last_worker_count.store(worker_count, std::memory_order_relaxed);
  last_reduction_fiber_chunks.store(schedule.chunks_per_fiber,
                                    std::memory_order_relaxed);
  return ExecuteStaticMapReduceWorkers(
      std::move(state), thread_pool, worker_count,
      [](State* worker_state, uint64_t worker_index) {
        ExecuteFiberPartialTasks(worker_state, worker_index);
      },
      [](State* worker_state) { MergeFiberPartials(worker_state); });
}

template <typename State>
void ExecuteGeneratedReductionTask(const State& state, uint64_t task) {
  const ParsedReductionPlan& plan = *state.execution;
  const uint64_t work_per_batch = plan.generated_work.size();
  const bool ordered_records =
      !state.grouped_output_owner && plan.records.size() != 1;
  const uint64_t batch =
      ordered_records ? task : task / work_per_batch;
  const auto* source_batch =
      plan.input_size == 0
          ? state.source
          : state.source + batch * plan.input_size;
  auto* result_batch =
      plan.output_size == 0
          ? state.result
          : state.result + batch * plan.output_size;
  if (ordered_records) {
    for (const ReductionRecord& record : plan.records) {
      state.bind_mapping(record, [&](auto scalar_op) {
        ExecuteReductionOutputRange<ReductionWriteMode::kAccumulateExisting>(
            record, source_batch, result_batch, 0, record.output_count,
            scalar_op, state.add_op);
      });
    }
    return;
  }
  const GeneratedReductionWork& work =
      plan.generated_work[task % work_per_batch];
  if (state.grouped_output_owner) {
    const ReductionGroup& group = plan.groups[work.owner_index];
    for (const std::size_t record_index : group.record_indices) {
      const auto& record = plan.records[record_index];
      state.bind_mapping(record, [&](auto scalar_op) {
        ExecuteReductionOutputRange<ReductionWriteMode::kAccumulateExisting>(
            record, source_batch, result_batch, work.output_start,
            work.output_count, scalar_op, state.add_op);
      });
    }
    return;
  }
  const auto& record = plan.records[work.owner_index];
  state.bind_mapping(record, [&](auto scalar_op) {
    ExecuteReductionOutputRange<ReductionWriteMode::kAssignReducedValue>(
        record, source_batch, result_batch, work.output_start, work.output_count,
        scalar_op, state.add_op);
  });
}

template <typename SourceElement, typename ResultElement, typename BindMapping,
          typename AddOp>
ffi::Future ExecuteGeneratedReductionPlan(
    std::shared_ptr<const ParsedReductionPlan> execution,
    const SourceElement* source, ResultElement* result,
    ffi::ThreadPool thread_pool, uint64_t available_workers,
    uint64_t batch_count, bool initialized_output,
    bool grouped_output_owner, BindMapping bind_mapping, AddOp add_op) {
  const ParsedReductionPlan& plan = *execution;
  uint64_t total_tasks = batch_count;
  const bool ordered_records =
      !grouped_output_owner && plan.records.size() != 1;
  if (!ordered_records &&
      !CheckedMultiply(batch_count, plan.generated_work.size(),
                       &total_tasks)) {
    return ErrorFuture(Invalid("generated reduction task count overflow"));
  }
  if (total_tasks == 0) {
    last_worker_count.store(initialized_output ? 1 : 0,
                            std::memory_order_relaxed);
    return ReadyFuture();
  }
  const bool threaded =
      (plan.strided_generated_mode & kGeneratedThreading) != 0;
  const uint64_t thread_budget =
      threaded ? std::max<uint64_t>(1, available_workers) : 1;
  const uint64_t worker_count = std::min(thread_budget, total_tasks);
  using State = GeneratedReductionState<SourceElement, ResultElement,
                                        BindMapping, AddOp>;
  auto state = std::make_shared<State>(State{
      std::move(execution), source, result, batch_count, worker_count,
      grouped_output_owner, std::move(bind_mapping), std::move(add_op)});
  if (worker_count <= 1) {
    last_worker_count.store(1, std::memory_order_relaxed);
    for (uint64_t task = 0; task < total_tasks; ++task) {
      ExecuteGeneratedReductionTask(*state, task);
    }
    return ReadyFuture();
  }

  last_worker_count.store(worker_count, std::memory_order_relaxed);
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule([state, completion, scheduled, total_tasks]() mutable {
        try {
          for (uint64_t task = scheduled; task < total_tasks;
               task += state->worker_count) {
            ExecuteGeneratedReductionTask(*state, task);
          }
          completion.CountDown();
        } catch (const std::exception&) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained generated reduction worker "
              "exception"));
        } catch (...) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained unknown generated reduction worker "
              "exception"));
        }
      });
    } catch (const std::exception&) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: failed to schedule generated reduction "
              "worker"));
      break;
    } catch (...) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: unknown generated reduction scheduling "
              "failure"));
      break;
    }
  }
  return future;
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename ScalarOp, typename AddOp>
ffi::Future ExecutePreparedStructuredReduction(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    uint64_t expected_input_dtype, uint64_t expected_output_dtype,
    uint64_t expected_mapped_dtype, uint64_t expected_scalar_policy,
    ScalarOp scalar_op, AddOp add_op) {
  if (prepared == nullptr || prepared->reduction_execution == nullptr) {
    return ErrorFuture(Invalid("prepared reduction state is unavailable"));
  }
  if (descriptor.size() != prepared->descriptor_bytes ||
      prepared->operation != PreparedOperation::kStructuredReduction) {
    return ErrorFuture(
        Invalid("prepared reduction descriptor/operation mismatch"));
  }
  const auto execution = prepared->reduction_execution;
  const ParsedReductionPlan& plan = *execution;
  if (plan.input_dtype != expected_input_dtype ||
      plan.output_dtype != expected_output_dtype ||
      plan.mapped_dtype != expected_mapped_dtype ||
      plan.scalar_policy != expected_scalar_policy) {
    return ErrorFuture(
        Invalid("reduction descriptor dtype does not match typed target"));
  }
  uint64_t batch_count = 0;
  ffi::Error error = ValidateBufferDimensions(
      source, result, plan.input_size, plan.output_size, &batch_count);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  uint64_t source_elements = 0;
  uint64_t result_elements = 0;
  if (!CheckedMultiply(batch_count, plan.input_size, &source_elements) ||
      !CheckedMultiply(batch_count, plan.output_size, &result_elements)) {
    return ErrorFuture(Invalid("reduction batched element count overflow"));
  }
  const uint64_t source_item_size = sizeof(*source.typed_data());
  const uint64_t result_item_size = sizeof(*result->typed_data());
  uint64_t source_bytes = 0;
  uint64_t result_bytes = 0;
  if (!CheckedMultiply(source_elements, source_item_size, &source_bytes) ||
      !CheckedMultiply(result_elements, result_item_size, &result_bytes)) {
    return ErrorFuture(Invalid("reduction batched byte count overflow"));
  }
  error = ValidateDisjointBuffers(source.typed_data(), result->typed_data(),
                                  source_bytes, result_bytes, 1);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }

  prepared_execute_count.fetch_add(1, std::memory_order_relaxed);
  native_call_count.fetch_add(1, std::memory_order_relaxed);
  const auto* source_data = source.typed_data();
  auto* result_data = result->typed_data();
  using ResultElement = std::remove_pointer_t<decltype(result_data)>;
  const int64_t pool_threads = thread_pool.num_threads();
  const uint64_t raw_available_workers =
      pool_threads > 0 ? static_cast<uint64_t>(pool_threads) : 0;
  last_available_worker_count.store(raw_available_workers,
                                    std::memory_order_relaxed);
  const uint64_t available_workers = std::min(
      raw_available_workers,
      worker_limit.load(std::memory_order_relaxed));
  const bool grouped_output_owner =
      ShouldUseGroupedOutputOwner(plan, batch_count);
  last_grouped_output_owner.store(grouped_output_owner,
                                  std::memory_order_relaxed);

  const auto bind_mapping = [scalar_op, expected_mapped_dtype](
      const ReductionRecord& record, auto execute) {
    BindScalarMapping<SourceDtype, ResultDtype>(
        record, expected_mapped_dtype, scalar_op, execute);
  };

  using GeneratedSourceElement =
      std::remove_const_t<std::remove_pointer_t<decltype(source_data)>>;
  last_reduction_fiber_chunks.store(0, std::memory_order_relaxed);
  const bool threaded =
      (plan.strided_generated_mode & kGeneratedThreading) != 0;
  const ReductionFiberSchedule fiber_schedule =
      SelectReductionFiberSchedule<GeneratedSourceElement, ResultElement>(
          plan, threaded ? available_workers : 1, batch_count);
  if (fiber_schedule.chunks_per_fiber != 0) {
    if (!fiber_schedule.complete_output && result_elements != 0) {
      std::fill(result_data, result_data + result_elements, ResultElement{});
    }
    return ExecuteFiberPartialReduction<GeneratedSourceElement, ResultElement>(
        execution, source_data, result_data, thread_pool, batch_count,
        fiber_schedule, bind_mapping, add_op);
  }
  const bool single_record_direct_assign = plan.records.size() == 1;
  const bool complete_single_record =
      single_record_direct_assign &&
      plan.records.front().reduction_count != 0 &&
      plan.records.front().output_count == plan.output_size;
  if (!complete_single_record && result_elements != 0) {
    std::fill(result_data, result_data + result_elements, ResultElement{});
  }
  return ExecuteGeneratedReductionPlan<GeneratedSourceElement, ResultElement>(
      execution, source_data, result_data, thread_pool, available_workers,
      batch_count, result_elements != 0, grouped_output_owner, bind_mapping,
      add_op);
}

#define TENSOR0_REDUCTION_IMPL(Name, SourceType, ResultType, InputCode,       \
                               OutputCode, MappedCode, ScalarPolicy,          \
                               ScalarType, AddType)                           \
  ffi::Future Name(                                                           \
      ffi::Span<const uint8_t> descriptor, PreparedState* prepared,           \
      ffi::Buffer<SourceType> source, ffi::ResultBuffer<ResultType> result,   \
      ffi::ThreadPool thread_pool) {                                          \
    return ContainExceptions([&] {                                            \
      return ExecutePreparedStructuredReduction<SourceType, ResultType>(     \
          descriptor, prepared, source, result, thread_pool, InputCode,       \
          OutputCode, MappedCode, ScalarPolicy, ScalarType{}, AddType{});     \
    });                                                                       \
  }

TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32Impl, ffi::F32,
                       ffi::F32, kDtypeF32, kDtypeF32,
                       kDtypeF32, kReductionScalarJaxTranspose,
                       F32Mapping, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16Impl, ffi::F16,
                       ffi::F16, kDtypeF16, kDtypeF16,
                       kDtypeF16, kReductionScalarJaxTranspose,
                       F16Mapping, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionBF16Impl, ffi::BF16,
                       ffi::BF16, kDtypeBF16, kDtypeBF16,
                       kDtypeBF16, kReductionScalarJaxTranspose,
                       BF16Mapping, AddBF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64Impl, ffi::C64,
                       ffi::C64, kDtypeC64, kDtypeC64,
                       kDtypeC64, kReductionScalarJaxTranspose,
                       C64Mapping, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32F16Impl, ffi::F32,
                       ffi::F16, kDtypeF32, kDtypeF16,
                       kDtypeF32, kReductionScalarJaxTranspose,
                       NarrowF16Product, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64F32Impl, ffi::C64,
                       ffi::F32, kDtypeC64, kDtypeF32,
                       kDtypeC64, kReductionScalarJaxTranspose,
                       ProjectedComplexProduct<float>, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32C64Impl, ffi::F32,
                       ffi::C64, kDtypeF32, kDtypeC64,
                       kDtypeF32, kReductionScalarJaxTranspose,
                       EmbeddedRealProduct<float>, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64Impl, ffi::F64,
                       ffi::F64, kDtypeF64, kDtypeF64,
                       kDtypeF64, kReductionScalarJaxTranspose,
                       F64Mapping, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128Impl, ffi::C128,
                       ffi::C128, kDtypeC128, kDtypeC128,
                       kDtypeC128, kReductionScalarJaxTranspose,
                       C128Mapping, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128F64Impl, ffi::C128,
                       ffi::F64, kDtypeC128, kDtypeF64,
                       kDtypeC128, kReductionScalarJaxTranspose,
                       ProjectedComplexProduct<double>, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64C128Impl, ffi::F64,
                       ffi::C128, kDtypeF64, kDtypeC128,
                       kDtypeF64, kReductionScalarJaxTranspose,
                       EmbeddedRealProduct<double>, AddC128)

TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32ForwardImpl, ffi::F32,
                       ffi::F32, kDtypeF32, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       F32Mapping, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16ForwardImpl, ffi::F16,
                       ffi::F16, kDtypeF16, kDtypeF16,
                       kDtypeF16, kReductionScalarForwardScaleCast,
                       F16Mapping, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionBF16ForwardImpl, ffi::BF16,
                       ffi::BF16, kDtypeBF16, kDtypeBF16,
                       kDtypeBF16, kReductionScalarForwardScaleCast,
                       BF16Mapping, AddBF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64ForwardImpl, ffi::C64,
                       ffi::C64, kDtypeC64, kDtypeC64,
                       kDtypeC64, kReductionScalarForwardScaleCast,
                       C64Mapping, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionPredForwardImpl, ffi::PRED,
                       ffi::PRED, kDtypePred, kDtypePred,
                       kDtypePred, kReductionScalarForwardScaleCast,
                       PREDMapping, AddPred)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS8ForwardImpl, ffi::S8,
                       ffi::S8, kDtypeS8, kDtypeS8,
                       kDtypeS8, kReductionScalarForwardScaleCast,
                       IntegerMapping<int8_t>, AddIntegerOp<int8_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS16ForwardImpl, ffi::S16,
                       ffi::S16, kDtypeS16, kDtypeS16,
                       kDtypeS16, kReductionScalarForwardScaleCast,
                       IntegerMapping<int16_t>, AddIntegerOp<int16_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS32ForwardImpl, ffi::S32,
                       ffi::S32, kDtypeS32, kDtypeS32,
                       kDtypeS32, kReductionScalarForwardScaleCast,
                       IntegerMapping<int32_t>, AddIntegerOp<int32_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS64ForwardImpl, ffi::S64,
                       ffi::S64, kDtypeS64, kDtypeS64,
                       kDtypeS64, kReductionScalarForwardScaleCast,
                       IntegerMapping<int64_t>, AddIntegerOp<int64_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU8ForwardImpl, ffi::U8,
                       ffi::U8, kDtypeU8, kDtypeU8,
                       kDtypeU8, kReductionScalarForwardScaleCast,
                       IntegerMapping<uint8_t>, AddIntegerOp<uint8_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU16ForwardImpl, ffi::U16,
                       ffi::U16, kDtypeU16, kDtypeU16,
                       kDtypeU16, kReductionScalarForwardScaleCast,
                       IntegerMapping<uint16_t>, AddIntegerOp<uint16_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU32ForwardImpl, ffi::U32,
                       ffi::U32, kDtypeU32, kDtypeU32,
                       kDtypeU32, kReductionScalarForwardScaleCast,
                       IntegerMapping<uint32_t>, AddIntegerOp<uint32_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU64ForwardImpl, ffi::U64,
                       ffi::U64, kDtypeU64, kDtypeU64,
                       kDtypeU64, kReductionScalarForwardScaleCast,
                       IntegerMapping<uint64_t>, AddIntegerOp<uint64_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16F32ForwardImpl, ffi::F16,
                       ffi::F32, kDtypeF16, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       F16F32Mapping, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32C64ForwardImpl, ffi::F32,
                       ffi::C64, kDtypeF32, kDtypeC64,
                       kDtypeC64, kReductionScalarForwardScaleCast,
                       F32C64Mapping, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64F32ForwardImpl, ffi::C64,
                       ffi::F32, kDtypeC64, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       C64F32Mapping, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64ForwardImpl, ffi::F64,
                       ffi::F64, kDtypeF64, kDtypeF64,
                       kDtypeF64, kReductionScalarForwardScaleCast,
                       F64Mapping, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128ForwardImpl, ffi::C128,
                       ffi::C128, kDtypeC128, kDtypeC128,
                       kDtypeC128, kReductionScalarForwardScaleCast,
                       C128Mapping, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64C128ForwardImpl, ffi::F64,
                       ffi::C128, kDtypeF64, kDtypeC128,
                       kDtypeC128, kReductionScalarForwardScaleCast,
                       F64C128Mapping, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128F64ForwardImpl, ffi::C128,
                       ffi::F64, kDtypeC128, kDtypeF64,
                       kDtypeF64, kReductionScalarForwardScaleCast,
                       C128F64Mapping, AddF64)

#undef TENSOR0_REDUCTION_IMPL

template <bool kConjugateLeft>
float DotProductValue(float left, float right) {
  return left * right;
}

template <bool kConjugateLeft>
std::complex<float> DotProductValue(std::complex<float> left,
                                    std::complex<float> right) {
  const float left_imaginary = kConjugateLeft ? -left.imag() : left.imag();
  return FullComplexMultiply(left.real(), left_imaginary, right.real(),
                             right.imag());
}

std::vector<AffineMapReduceSubdomain> BuildDotSubdomains(
    const ParsedDotPlan& plan, uint64_t desired_tasks) {
  std::vector<AffineMapReduceSubdomain> domains;
  domains.reserve(std::max<uint64_t>(desired_tasks, plan.records.size()));
  for (std::size_t record_index = 0; record_index < plan.records.size();
       ++record_index) {
    const Record& record = plan.records[record_index];
    if (record.logical_elements == 0) {
      continue;
    }
    AffineMapReduceSubdomain domain;
    domain.record_index = record_index;
    domain.operand_offsets = {record.source_offset,
                              record.destination_offset};
    domain.logical_elements = record.logical_elements;
    domain.shape = record.shape;
    domains.push_back(domain);
  }
  SplitAffineMapReduceSubdomains(plan.records, desired_tasks, &domains);
  return domains;
}

template <bool kConjugateLeft, typename Element, typename AddOp>
Element ExecuteDotSubdomain(const Record& record,
                            const AffineMapReduceSubdomain& domain,
                            const Element* left, const Element* right,
                            AddOp add_op) {
  const auto map_op = [left, right](int64_t left_address,
                                     int64_t right_address) {
    return DotProductValue<kConjugateLeft>(left[left_address],
                                            right[right_address]);
  };
  return ExecuteAffineMapReduceSubdomain<Record, Element>(record, domain,
                                                           map_op, add_op);
}

template <bool kConjugateLeft, typename Element, typename AddOp>
Element ExecuteDotRecord(const Record& record, const Element* left,
                         const Element* right, AddOp add_op) {
  AffineMapReduceSubdomain domain;
  domain.operand_offsets = {record.source_offset,
                            record.destination_offset};
  domain.logical_elements = record.logical_elements;
  domain.shape = record.shape;
  return ExecuteDotSubdomain<kConjugateLeft>(record, domain, left, right,
                                              add_op);
}

template <typename Element>
struct DotExecutionState {
  std::shared_ptr<const ParsedDotPlan> execution;
  const Element* left = nullptr;
  const Element* right = nullptr;
  Element* result = nullptr;
  uint64_t batch_count = 0;
  std::vector<AffineMapReduceSubdomain> domains;
  std::vector<PaddedReductionPartial<Element>> partials;
  std::atomic<uint64_t> remaining_workers{0};
  std::atomic<bool> failed{false};
};

template <bool kConjugateLeft, typename Element, typename AddOp>
void ExecuteDotBatchTasks(DotExecutionState<Element>* state,
                          uint64_t worker_index, uint64_t worker_count,
                          AddOp add_op) {
  const ParsedDotPlan& plan = *state->execution;
  for (uint64_t batch = worker_index; batch < state->batch_count;
       batch += worker_count) {
    const auto* left_batch =
        plan.left_size == 0 ? state->left
                            : state->left + batch * plan.left_size;
    const auto* right_batch =
        plan.right_size == 0 ? state->right
                             : state->right + batch * plan.right_size;
    Element accumulator{};
    for (const Record& record : plan.records) {
      accumulator = add_op(
          accumulator,
          ExecuteDotRecord<kConjugateLeft>(record, left_batch, right_batch,
                                            add_op));
    }
    state->result[batch] = accumulator;
  }
}

template <bool kConjugateLeft, typename Element, typename AddOp>
void ExecuteDotPartialTasks(DotExecutionState<Element>* state,
                            uint64_t worker_index, uint64_t worker_count,
                            AddOp add_op) {
  const ParsedDotPlan& plan = *state->execution;
  for (uint64_t task = worker_index; task < state->domains.size();
       task += worker_count) {
    const AffineMapReduceSubdomain& domain = state->domains[task];
    state->partials[task].value = ExecuteDotSubdomain<kConjugateLeft>(
        plan.records[domain.record_index], domain, state->left, state->right,
        add_op);
  }
}

template <typename Element, typename AddOp>
void MergeDotPartials(DotExecutionState<Element>* state, AddOp add_op) {
  Element accumulator{};
  for (const auto& partial : state->partials) {
    accumulator = add_op(accumulator, partial.value);
  }
  state->result[0] = accumulator;
}

template <bool kConjugateLeft, typename Element, typename AddOp>
ffi::Future ExecuteThreadedDot(
    std::shared_ptr<const ParsedDotPlan> execution, const Element* left,
    const Element* right, Element* result, ffi::ThreadPool thread_pool,
    uint64_t batch_count, uint64_t worker_count,
    std::vector<AffineMapReduceSubdomain> domains, AddOp add_op) {
  auto state = std::make_shared<DotExecutionState<Element>>();
  state->execution = std::move(execution);
  state->left = left;
  state->right = right;
  state->result = result;
  state->batch_count = batch_count;
  state->domains = std::move(domains);
  state->partials.resize(state->domains.size());
  state->remaining_workers.store(worker_count, std::memory_order_relaxed);
  last_worker_count.store(worker_count, std::memory_order_relaxed);
  last_reduction_fiber_chunks.store(state->domains.empty()
                                        ? 0
                                        : state->domains.size(),
                                    std::memory_order_relaxed);
  const bool partial = batch_count == 1 && !state->domains.empty();
  return ExecuteStaticMapReduceWorkers(
      std::move(state), thread_pool, worker_count,
      [partial, worker_count, add_op](DotExecutionState<Element>* worker_state,
                                      uint64_t worker_index) {
        if (partial) {
          ExecuteDotPartialTasks<kConjugateLeft>(
              worker_state, worker_index, worker_count, add_op);
        } else {
          ExecuteDotBatchTasks<kConjugateLeft>(
              worker_state, worker_index, worker_count, add_op);
        }
      },
      [partial, add_op](DotExecutionState<Element>* worker_state) {
        if (partial) {
          MergeDotPartials(worker_state, add_op);
        }
      });
}

template <ffi::DataType Dtype, bool kConjugateLeft, typename AddOp>
ffi::Future ExecutePreparedDot(ffi::Span<const uint8_t> descriptor,
                               PreparedState* prepared,
                               ffi::Buffer<Dtype> left,
                               ffi::Buffer<Dtype> right,
                               ffi::ResultBuffer<Dtype> result,
                               ffi::ThreadPool thread_pool,
                               uint64_t expected_dtype, AddOp add_op) {
  if (prepared == nullptr || prepared->dot_execution == nullptr) {
    return ErrorFuture(Invalid("prepared dot state is unavailable"));
  }
  if (descriptor.size() != prepared->descriptor_bytes ||
      prepared->operation != PreparedOperation::kDot) {
    return ErrorFuture(Invalid("prepared dot descriptor/operation mismatch"));
  }
  const auto execution = prepared->dot_execution;
  const ParsedDotPlan& plan = *execution;
  if (plan.dtype != expected_dtype ||
      plan.conjugate_left != kConjugateLeft) {
    return ErrorFuture(Invalid("dot descriptor does not match typed target"));
  }
  uint64_t batch_count = 0;
  ffi::Error error = ValidateBufferDimensions(
      left, result, plan.left_size, 1, &batch_count, true);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  uint64_t right_elements = 0;
  error = CountBufferElements(right.dimensions(), &right_elements);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  uint64_t expected_right_elements = 0;
  uint64_t left_elements = 0;
  if (!CheckedMultiply(batch_count, plan.right_size,
                       &expected_right_elements) ||
      !CheckedMultiply(batch_count, plan.left_size, &left_elements)) {
    return ErrorFuture(Invalid("dot batched element count overflow"));
  }
  if (right_elements != expected_right_elements) {
    return ErrorFuture(Invalid("dot right batch count does not match"));
  }
  const uint64_t item_size = sizeof(*result->typed_data());
  error = ValidateDisjointBuffers(left.typed_data(), result->typed_data(),
                                  left_elements, batch_count, item_size);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  error = ValidateDisjointBuffers(right.typed_data(), result->typed_data(),
                                  right_elements, batch_count, item_size);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }

  prepared_execute_count.fetch_add(1, std::memory_order_relaxed);
  native_call_count.fetch_add(1, std::memory_order_relaxed);
  const int64_t pool_threads = thread_pool.num_threads();
  const uint64_t raw_available_workers =
      pool_threads > 0 ? static_cast<uint64_t>(pool_threads) : 0;
  last_available_worker_count.store(raw_available_workers,
                                    std::memory_order_relaxed);
  const uint64_t available_workers = std::min(
      raw_available_workers,
      worker_limit.load(std::memory_order_relaxed));
  last_reduction_fiber_chunks.store(0, std::memory_order_relaxed);
  const auto* left_data = left.typed_data();
  const auto* right_data = right.typed_data();
  auto* result_data = result->typed_data();
  using Element = std::remove_pointer_t<decltype(result_data)>;
  uint64_t logical_elements = 0;
  for (const Record& record : plan.records) {
    if (!CheckedAdd(logical_elements, record.logical_elements,
                    &logical_elements)) {
      return ErrorFuture(Invalid("dot logical element count overflow"));
    }
  }
  uint64_t traffic_elements = 0;
  uint64_t traffic_bytes = 0;
  if (!CheckedMultiply(logical_elements, 2, &traffic_elements) ||
      !CheckedMultiply(traffic_elements, item_size, &traffic_bytes)) {
    return ErrorFuture(Invalid("dot traffic byte count overflow"));
  }
  if (batch_count > 1 && available_workers > 1) {
    uint64_t batched_traffic_bytes = 0;
    if (!CheckedMultiply(batch_count, traffic_bytes,
                         &batched_traffic_bytes)) {
      return ErrorFuture(Invalid("batched dot traffic byte count overflow"));
    }
    if (batched_traffic_bytes >= kReductionParallelMinimumBytes) {
      return ExecuteThreadedDot<kConjugateLeft>(
          execution, left_data, right_data, result_data, thread_pool,
          batch_count, std::min(available_workers, batch_count), {}, add_op);
    }
  }
  if (batch_count == 1 && available_workers > 1 &&
      traffic_bytes >= kReductionParallelMinimumBytes) {
    const uint64_t desired_tasks =
        std::min(kReductionFiberMaximumChunks, available_workers);
    auto domains = BuildDotSubdomains(plan, desired_tasks);
    uint64_t scratch_bytes = 0;
    if (!CheckedMultiply(domains.size(),
                         sizeof(PaddedReductionPartial<Element>),
                         &scratch_bytes) ||
        scratch_bytes > kReductionFiberMaximumScratchBytes) {
      return ErrorFuture(Invalid("dot partial scratch exceeds native cap"));
    }
    if (domains.size() > 1) {
      const uint64_t worker_count =
          std::min<uint64_t>(available_workers, domains.size());
      return ExecuteThreadedDot<kConjugateLeft>(
          execution, left_data, right_data, result_data, thread_pool,
          batch_count, worker_count, std::move(domains), add_op);
    }
  }
  last_worker_count.store(batch_count == 0 ? 0 : 1,
                          std::memory_order_relaxed);
  for (uint64_t batch = 0; batch < batch_count; ++batch) {
    const auto* left_batch =
        plan.left_size == 0 ? left_data : left_data + batch * plan.left_size;
    const auto* right_batch = plan.right_size == 0
                                  ? right_data
                                  : right_data + batch * plan.right_size;
    Element accumulator{};
    for (const Record& record : plan.records) {
      accumulator = add_op(
          accumulator,
          ExecuteDotRecord<kConjugateLeft>(record, left_batch, right_batch,
                                            add_op));
    }
    result_data[batch] = accumulator;
  }
  return ReadyFuture();
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedDotImpl(
    ffi::Span<const uint8_t> descriptor) {
  try {
    return InstantiatePreparedDot(descriptor);
  } catch (const std::exception&) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained dot instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown dot instantiate exception"));
  }
}

#define TENSOR0_DOT_IMPL(Name, Dtype, Conjugate, Code, Add)            \
  ffi::Future Name(ffi::Span<const uint8_t> descriptor,               \
                   PreparedState* prepared, ffi::Buffer<Dtype> left,  \
                   ffi::Buffer<Dtype> right,                          \
                   ffi::ResultBuffer<Dtype> result,                   \
                   ffi::ThreadPool thread_pool) {                      \
    return ExecutePreparedDot<Dtype, Conjugate>(                       \
        descriptor, prepared, left, right, result, thread_pool, Code,  \
        Add{});                                                        \
  }

TENSOR0_DOT_IMPL(ExecuteDotuF32Impl, ffi::F32, false, kDtypeF32, AddF32)
TENSOR0_DOT_IMPL(ExecuteDotcF32Impl, ffi::F32, true, kDtypeF32, AddF32)
TENSOR0_DOT_IMPL(ExecuteDotuC64Impl, ffi::C64, false, kDtypeC64, AddC64)
TENSOR0_DOT_IMPL(ExecuteDotcC64Impl, ffi::C64, true, kDtypeC64, AddC64)

#undef TENSOR0_DOT_IMPL
// Structured-reduction scalar policies, domain walkers, and typed execution.
