template <ffi::DataType SourceDtype, ffi::DataType ResultDtype, typename Map>
void ExecuteForwardMapLeaf(const Record& record, const ffi::NativeType<SourceDtype>* input,
                           ffi::NativeType<ResultDtype>* output, LeafKernel kind,
                           uint64_t unit_start, uint64_t unit_count, Map map,
                           uint64_t expected_dtype = DescriptorDtypeCode<ResultDtype>()) {
  BindScalarMapping<SourceDtype, ResultDtype>(record, expected_dtype, map, [&](auto bound) {
    ExecuteAffineLeafRange(record, input, output, kind, unit_start,
        unit_count, bound, AssignMappedValue{});
  });
}

enum class PreparedOperation : uint8_t {
  kFreshMap,
  kSelectedScale,
  kAxpby,
  kStructuredReduction,
  kDot,
};

bool GeneratedPolicyFeasibilitySupported(const ParsedPlan& plan,
                                         PreparedOperation operation) {
  if (operation == PreparedOperation::kSelectedScale) {
    return false;
  }
  if (operation == PreparedOperation::kFreshMap) {
    return plan.scalar_kind == kScalarForwardScaleCast ||
           plan.scalar_kind == kScalarJaxTranspose;
  }
  return operation == PreparedOperation::kAxpby &&
         plan.scalar_kind == kScalarForwardScaleCast;
}

bool ProductionGeneratedAffineEligible(const PreparedAffinePlan& execution) {
  const ParsedPlan& plan = execution.plan;
  if (plan.records.size() == 1 && execution.work.size() == 1 &&
      plan.coverage == kCoverageCompleteUnique &&
      plan.source_size == plan.output_size &&
      plan.copied_elements == plan.source_size &&
      plan.records[0].source_offset == 0 &&
      plan.records[0].destination_offset == 0 &&
      IsCompactSameMapping(plan.records[0])) {
    return false;
  }
  return std::all_of(
      execution.work.begin(), execution.work.end(),
      [](const RecordWork& work) {
        return work.leaf_kernel == LeafKernel::kContiguousRow ||
               work.leaf_kernel == LeafKernel::kBroadcastContiguousRow ||
               work.leaf_kernel == LeafKernel::kScalarRow ||
               work.leaf_kernel == LeafKernel::kContiguousInnerRows;
      });
}

uint64_t TrackedExecutionDataBytes(const PreparedAffinePlan& data) {
  return sizeof(PreparedAffinePlan) +
         data.plan.records.capacity() * sizeof(Record) +
         data.work.capacity() * sizeof(RecordWork) +
         data.strided_generated_records.capacity() *
             sizeof(GeneratedRecordProgram);
}

struct UpdateRecordTypes {
  std::array<uint64_t, 7> stages{};
};

struct PreparedState {
  static ffi::TypeId id;

  PreparedState(std::shared_ptr<const PreparedAffinePlan> value,
                uint64_t descriptor_size, uint64_t state_size,
                PreparedOperation prepared_operation)
      : execution(std::move(value)),
        descriptor_bytes(descriptor_size),
        tracked_bytes(state_size),
        operation(prepared_operation) {
    prepared_live_state_count.fetch_add(1, std::memory_order_relaxed);
    prepared_live_bytes.fetch_add(tracked_bytes, std::memory_order_relaxed);
    prepared_last_state_bytes.store(tracked_bytes,
                                    std::memory_order_relaxed);
    prepared_last_descriptor_bytes.store(descriptor_bytes,
                                         std::memory_order_relaxed);
  }

  PreparedState(std::shared_ptr<const ParsedReductionPlan> value,
                uint64_t descriptor_size, uint64_t state_size)
      : reduction_execution(std::move(value)),
        descriptor_bytes(descriptor_size),
        tracked_bytes(state_size),
        operation(PreparedOperation::kStructuredReduction) {
    prepared_live_state_count.fetch_add(1, std::memory_order_relaxed);
    prepared_live_bytes.fetch_add(tracked_bytes, std::memory_order_relaxed);
    prepared_last_state_bytes.store(tracked_bytes,
                                    std::memory_order_relaxed);
    prepared_last_descriptor_bytes.store(descriptor_bytes,
                                         std::memory_order_relaxed);
  }

  PreparedState(std::shared_ptr<const ParsedDotPlan> value,
                uint64_t descriptor_size, uint64_t state_size)
      : dot_execution(std::move(value)),
        descriptor_bytes(descriptor_size),
        tracked_bytes(state_size),
        operation(PreparedOperation::kDot) {
    prepared_live_state_count.fetch_add(1, std::memory_order_relaxed);
    prepared_live_bytes.fetch_add(tracked_bytes, std::memory_order_relaxed);
    prepared_last_state_bytes.store(tracked_bytes,
                                    std::memory_order_relaxed);
    prepared_last_descriptor_bytes.store(descriptor_bytes,
                                         std::memory_order_relaxed);
  }

  ~PreparedState() {
    prepared_live_state_count.fetch_sub(1, std::memory_order_relaxed);
    prepared_live_bytes.fetch_sub(tracked_bytes, std::memory_order_relaxed);
    prepared_destroyed_state_count.fetch_add(1,
                                             std::memory_order_relaxed);
  }

  std::shared_ptr<const PreparedAffinePlan> execution;
  std::shared_ptr<const ParsedReductionPlan> reduction_execution;
  std::shared_ptr<const ParsedDotPlan> dot_execution;
  std::vector<UpdateRecordTypes> update_types;
  uint64_t descriptor_bytes = 0;
  uint64_t tracked_bytes = 0;
  PreparedOperation operation = PreparedOperation::kFreshMap;
};

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();

ffi::ErrorOr<std::unique_ptr<PreparedState>> PublishPrepared(
    std::shared_ptr<PreparedAffinePlan> execution,
    uint64_t descriptor_size, PreparedOperation operation) {
  constexpr uint64_t kPreparedBaseBytes = 4 * 1024;
  constexpr uint64_t kPreparedBytesPerRecord = 512;
  constexpr uint64_t kPreparedAbsoluteBytes = 4 * 1024 * 1024;
  uint64_t record_bytes = 0;
  uint64_t relative_limit = 0;
  if (!CheckedMultiply(execution->plan.records.size(),
                       kPreparedBytesPerRecord, &record_bytes) ||
      !CheckedAdd(kPreparedBaseBytes, record_bytes, &relative_limit)) {
    return ffi::Unexpected(Invalid("prepared state size limit overflow"));
  }
  const uint64_t size_limit =
      std::min(relative_limit, kPreparedAbsoluteBytes);
  std::vector<UpdateRecordTypes> update_types;
  if (operation == PreparedOperation::kAxpby) {
    update_types.resize(execution->plan.records.size());
  }
  const uint64_t type_bytes = update_types.capacity() * sizeof(UpdateRecordTypes);
  uint64_t tracked_bytes =
      sizeof(PreparedState) + TrackedExecutionDataBytes(*execution) + type_bytes;
  if (tracked_bytes > size_limit &&
      !execution->strided_generated_records.empty()) {
    execution->strided_generated_mode = 0;
    std::vector<GeneratedRecordProgram>().swap(
        execution->strided_generated_records);
    tracked_bytes =
        sizeof(PreparedState) + TrackedExecutionDataBytes(*execution) + type_bytes;
  }
  if (tracked_bytes > size_limit) {
    return ffi::Unexpected(Invalid("prepared state exceeds size limit"));
  }
  auto state = std::make_unique<PreparedState>(
      std::move(execution), descriptor_size, tracked_bytes, operation);
  state->update_types = std::move(update_types);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> PublishPreparedReduction(
    std::shared_ptr<ParsedReductionPlan> execution,
    uint64_t descriptor_size) {
  const uint64_t tracked_bytes =
      sizeof(PreparedState) + TrackedReductionPlanBytes(*execution);
  constexpr uint64_t kPreparedBaseBytes = 4 * 1024;
  constexpr uint64_t kPreparedBytesPerRecord = 1024;
  constexpr uint64_t kPreparedAbsoluteBytes = 4 * 1024 * 1024;
  uint64_t record_bytes = 0;
  uint64_t relative_limit = 0;
  if (!CheckedMultiply(execution->records.size(),
                       kPreparedBytesPerRecord, &record_bytes) ||
      !CheckedAdd(kPreparedBaseBytes, record_bytes, &relative_limit)) {
    return ffi::Unexpected(
        Invalid("prepared reduction state size limit overflow"));
  }
  if (tracked_bytes > std::min(relative_limit, kPreparedAbsoluteBytes)) {
    return ffi::Unexpected(
        Invalid("prepared reduction state exceeds size limit"));
  }
  return std::make_unique<PreparedState>(
      std::move(execution), descriptor_size, tracked_bytes);
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> PublishPreparedDot(
    std::shared_ptr<ParsedDotPlan> execution, uint64_t descriptor_size) {
  const uint64_t tracked_bytes =
      sizeof(PreparedState) + TrackedDotPlanBytes(*execution);
  constexpr uint64_t kPreparedBaseBytes = 4 * 1024;
  constexpr uint64_t kPreparedBytesPerRecord = 512;
  constexpr uint64_t kPreparedAbsoluteBytes = 4 * 1024 * 1024;
  uint64_t record_bytes = 0;
  uint64_t relative_limit = 0;
  if (!CheckedMultiply(execution->records.size(), kPreparedBytesPerRecord,
                       &record_bytes) ||
      !CheckedAdd(kPreparedBaseBytes, record_bytes, &relative_limit)) {
    return ffi::Unexpected(Invalid("prepared dot state size limit overflow"));
  }
  if (tracked_bytes > std::min(relative_limit, kPreparedAbsoluteBytes)) {
    return ffi::Unexpected(Invalid("prepared dot state exceeds size limit"));
  }
  return std::make_unique<PreparedState>(std::move(execution),
                                         descriptor_size, tracked_bytes);
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(
    ffi::Span<const uint8_t> descriptor, PreparedOperation operation) {
  prepared_instantiate_count.fetch_add(1, std::memory_order_relaxed);
  auto execution = std::make_shared<PreparedAffinePlan>();
  ffi::Error error = CompileAffinePlan(
      descriptor, &execution->plan, &execution->work,
      &execution->chunks_per_batch,
      &execution->parallel_minimum_bytes,
      operation == PreparedOperation::kSelectedScale);
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  if (force_generic_for_tests.load(std::memory_order_relaxed)) {
    std::vector<RecordWork> generic_work;
    uint64_t generic_chunks_per_batch = 0;
    error = CompileRecordWork(
        execution->plan, DtypeItemSize(execution->plan.dtype),
        &generic_work, &generic_chunks_per_batch, true);
    if (!error.success()) {
      return ffi::Unexpected(std::move(error));
    }
    execution->work = std::move(generic_work);
    execution->chunks_per_batch = generic_chunks_per_batch;
    execution->parallel_minimum_bytes = kParallelMappedMinimumWorkBytes;
    execution->force_generic = true;
  } else if (force_generated_baseline_for_tests.load(
                 std::memory_order_relaxed)) {
    std::vector<RecordWork> generated_work;
    uint64_t generated_chunks_per_batch = 0;
    error = CompileRecordWork(
        execution->plan, DtypeItemSize(execution->plan.dtype),
        &generated_work, &generated_chunks_per_batch, false, true);
    if (!error.success()) {
      return ffi::Unexpected(std::move(error));
    }
    execution->work = std::move(generated_work);
    execution->chunks_per_batch = generated_chunks_per_batch;
  }
  const uint64_t generated_mode =
      execution->force_generic || !ProductionGeneratedAffineEligible(*execution)
          ? 0
          : kGeneratedProductionMode;
  const bool generated_ranks_supported = std::all_of(
      execution->plan.records.begin(), execution->plan.records.end(),
      [](const Record& record) {
        return record.rank <= kMaximumRank;
      });
  if ((generated_mode & kGeneratedEnable) != 0 &&
      GeneratedPolicyFeasibilitySupported(execution->plan, operation) &&
      !execution->plan.records.empty() && generated_ranks_supported) {
    execution->strided_generated_mode = generated_mode;
    execution->strided_generated_records.reserve(
        execution->plan.records.size());
    for (const Record& record : execution->plan.records) {
      execution->strided_generated_records.push_back(CompileGeneratedRecord(
          record, generated_mode,
          DtypeItemSize(execution->plan.source_dtype),
          DtypeItemSize(execution->plan.result_dtype)));
    }
  }
  if (operation != PreparedOperation::kFreshMap &&
      execution->plan.scalar_kind != kScalarForwardScaleCast) {
    return ffi::Unexpected(
        Invalid("prepared update operation requires forward scalar policy"));
  }
  if (operation == PreparedOperation::kSelectedScale) {
    const ParsedPlan& plan = execution->plan;
    if (plan.source_size != plan.output_size) {
      return ffi::Unexpected(
          Invalid("selected scale requires equal storage sizes"));
    }
    for (const Record& record : plan.records) {
      if (record.source_offset != record.destination_offset ||
          record.source_strides != record.destination_strides) {
        return ffi::Unexpected(
            Invalid("selected scale requires identical affine addresses"));
      }
      if (!record.identity_scale) {
        return ffi::Unexpected(
            Invalid("selected scale requires static unit scale"));
      }
    }
  }
  return PublishPrepared(std::move(execution), descriptor.size(), operation);
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedReduction(
    ffi::Span<const uint8_t> descriptor, bool grouped_output_owner = false) {
  prepared_instantiate_count.fetch_add(1, std::memory_order_relaxed);
  auto execution = std::make_shared<ParsedReductionPlan>();
  ffi::Error error = ParseReductionDescriptor(descriptor, execution.get());
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  error = BuildExactReductionGroups(execution.get(), grouped_output_owner);
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  execution->strided_generated_mode = kGeneratedProductionMode;
  error = CompileGeneratedReductionWork(execution.get());
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  return PublishPreparedReduction(std::move(execution), descriptor.size());
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedDot(
    ffi::Span<const uint8_t> descriptor) {
  prepared_instantiate_count.fetch_add(1, std::memory_order_relaxed);
  auto execution = std::make_shared<ParsedDotPlan>();
  ffi::Error error = ParseDotDescriptor(descriptor, execution.get());
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  return PublishPreparedDot(std::move(execution), descriptor.size());
}

template <typename SourceElement, typename ResultElement,
          typename RangeFunction>
struct PreparedParallelState {
  PreparedParallelState(
      std::shared_ptr<const PreparedAffinePlan> execution_data,
      const SourceElement* source_data, ResultElement* result_data,
      uint64_t batches, uint64_t total_chunks_value, uint64_t workers,
      RangeFunction range_function)
      : execution(std::move(execution_data)),
        source(source_data),
        result(result_data),
        batch_count(batches),
        total_chunks(total_chunks_value),
        worker_count(workers),
        execute_range(std::move(range_function)) {}

  std::shared_ptr<const PreparedAffinePlan> execution;
  const SourceElement* source = nullptr;
  ResultElement* result = nullptr;
  uint64_t batch_count = 0;
  uint64_t total_chunks = 0;
  uint64_t worker_count = 0;
  RangeFunction execute_range;
};

template <typename State>
void ExecutePreparedParallelWorker(const std::shared_ptr<State>& state,
                                   uint64_t worker_index) {
  const ParsedPlan& plan = state->execution->plan;
  const auto& work = state->execution->work;
  const uint64_t chunks_per_batch = state->execution->chunks_per_batch;
  for (uint64_t global_chunk = worker_index;
       global_chunk < state->total_chunks;
       global_chunk += state->worker_count) {
    const uint64_t batch = global_chunk / chunks_per_batch;
    const uint64_t local_chunk = global_chunk - batch * chunks_per_batch;
    const auto item = std::upper_bound(
        work.begin(), work.end(), local_chunk,
        [](uint64_t chunk, const RecordWork& candidate) {
          return chunk < candidate.chunk_begin + candidate.chunk_count;
        });
    if (item == work.end()) {
      return;
    }
    const uint64_t chunk_in_record = local_chunk - item->chunk_begin;
    uint64_t unit_start = chunk_in_record * item->units_per_chunk;
    uint64_t unit_count = std::min(
        item->units_per_chunk, item->unit_count - unit_start);
    if (item->leaf_kernel == LeafKernel::kScalarRow) {
      unit_start *= item->elements_per_unit;
      unit_count *= item->elements_per_unit;
    }
    const auto* source_batch =
        plan.source_size == 0
            ? state->source
            : state->source + batch * plan.source_size;
    auto* result_batch =
        plan.output_size == 0
            ? state->result
            : state->result + batch * plan.output_size;
    state->execute_range(plan.records[item->record_index], source_batch,
                         result_batch, item->leaf_kernel, unit_start,
                         unit_count);
  }
}

struct GeneratedRangeParallelState {
  GeneratedRangeParallelState(
      std::vector<GeneratedRecordProgram> values,
      const uint8_t* source_data, uint8_t* result_data,
      uint64_t source_bytes_per_batch,
      uint64_t result_bytes_per_batch, uint64_t batches,
      bool records_are_ordered, uint64_t workers,
      std::shared_ptr<const void> range_context_value,
      void (*range_function)(const void*, const Record&, const void*, void*,
                             LeafKernel, uint64_t, uint64_t))
      : programs(std::move(values)),
        source(source_data),
        result(result_data),
        source_batch_bytes(source_bytes_per_batch),
        result_batch_bytes(result_bytes_per_batch),
        batch_count(batches),
        ordered_records(records_are_ordered),
        worker_count(workers),
        range_context(std::move(range_context_value)),
        execute_range(std::move(range_function)) {}

  std::vector<GeneratedRecordProgram> programs;
  const uint8_t* source = nullptr;
  uint8_t* result = nullptr;
  uint64_t source_batch_bytes = 0;
  uint64_t result_batch_bytes = 0;
  uint64_t batch_count = 0;
  bool ordered_records = false;
  uint64_t worker_count = 0;
  std::shared_ptr<const void> range_context;
  void (*execute_range)(const void*, const Record&, const void*, void*,
                        LeafKernel, uint64_t, uint64_t) = nullptr;
};

void ExecuteGeneratedRangeRecord(const GeneratedRecordProgram& program,
                                 const void* source, void* result,
                                 const GeneratedRangeParallelState& state) {
  ForEachGeneratedBlock(program, [&](const Record& block) {
    const LeafKernel leaf =
        IsCompactSameMapping(block)
            ? LeafKernel::kContiguousRow
            : (IsBroadcastContiguousRowCandidate(block)
                   ? LeafKernel::kBroadcastContiguousRow
                   : (IsContiguousInnerRowsCandidate(block)
                          ? LeafKernel::kContiguousInnerRows
                          : LeafKernel::kScalarRow));
    const uint64_t unit_count =
        leaf == LeafKernel::kContiguousInnerRows ||
                leaf == LeafKernel::kBroadcastContiguousRow
            ? GenericRowCount(block)
            : block.logical_elements;
    state.execute_range(state.range_context.get(), block, source, result,
                        leaf, 0, unit_count);
  });
}

void ExecuteGeneratedRangeBatchTask(
    const GeneratedRangeParallelState& state, uint64_t task) {
  const uint64_t programs_per_batch = state.programs.size();
  const uint64_t batch =
      state.ordered_records ? task : task / programs_per_batch;
  const uint8_t* source_batch =
      state.source_batch_bytes == 0
          ? state.source
          : state.source + batch * state.source_batch_bytes;
  uint8_t* result_batch =
      state.result_batch_bytes == 0
          ? state.result
          : state.result + batch * state.result_batch_bytes;
  if (state.ordered_records) {
    for (const GeneratedRecordProgram& program : state.programs) {
      ExecuteGeneratedRangeRecord(program, source_batch, result_batch, state);
    }
    return;
  }
  ExecuteGeneratedRangeRecord(state.programs[task % programs_per_batch],
                              source_batch, result_batch, state);
}

ffi::Future ExecuteStridedGeneratedErasedProgram(
    const PreparedAffinePlan& execution, const void* source, void* result,
    uint64_t source_item_size, uint64_t result_item_size,
    ffi::ThreadPool thread_pool, uint64_t available_workers,
    uint64_t batch_count, std::shared_ptr<const void> range_context,
    void (*execute_range)(const void*, const Record&, const void*, void*,
                          LeafKernel, uint64_t, uint64_t)) {
  const uint64_t mode = execution.strided_generated_mode;
  const bool threaded = (mode & kGeneratedThreading) != 0;
  const uint64_t thread_budget =
      threaded ? std::max<uint64_t>(1, available_workers) : 1;
  const bool multiple_records =
      execution.strided_generated_records.size() > 1;
  uint64_t record_elements = 0;
  for (const GeneratedRecordProgram& program :
       execution.strided_generated_records) {
    record_elements =
        std::numeric_limits<uint64_t>::max() - record_elements <
                program.record.logical_elements
            ? std::numeric_limits<uint64_t>::max()
            : record_elements + program.record.logical_elements;
  }
  constexpr uint64_t kMinimumRecordTaskElements = UINT64_C(1) << 15;
  const bool parallel_records =
      multiple_records && thread_budget > 1 &&
      record_elements / thread_budget >= kMinimumRecordTaskElements;
  // Affine plans require cross-record destination uniqueness. Large record
  // sequences can therefore use independent worker ownership, while small
  // sequences remain ordered to avoid thread-pool overhead.
  const bool ordered_records = multiple_records && !parallel_records;
  std::vector<GeneratedRecordProgram> programs;
  if (multiple_records) {
    programs = execution.strided_generated_records;
  } else {
    const uint64_t active_batches = std::min(batch_count, thread_budget);
    const uint64_t subdomain_budget =
        std::max<uint64_t>(1, thread_budget / active_batches);
    programs.reserve(subdomain_budget);
    AppendGeneratedSubdomains(execution.strided_generated_records.front(),
                              subdomain_budget, &programs);
  }
  uint64_t total_tasks = batch_count;
  if (!ordered_records &&
      !CheckedMultiply(batch_count, programs.size(), &total_tasks)) {
    return ErrorFuture(
        Invalid("generated range batched task count overflow"));
  }
  const uint64_t worker_count = std::min(thread_budget, total_tasks);
  uint64_t source_batch_bytes = 0;
  uint64_t result_batch_bytes = 0;
  if (!CheckedMultiply(execution.plan.source_size, source_item_size,
                       &source_batch_bytes) ||
      !CheckedMultiply(execution.plan.output_size, result_item_size,
                       &result_batch_bytes)) {
    return ErrorFuture(Invalid("generated range batch byte count overflow"));
  }
  auto state = std::make_shared<GeneratedRangeParallelState>(
      std::move(programs), static_cast<const uint8_t*>(source),
      static_cast<uint8_t*>(result), source_batch_bytes,
      result_batch_bytes, batch_count, ordered_records, worker_count,
      std::move(range_context), execute_range);
  if (worker_count <= 1) {
    last_worker_count.store(1, std::memory_order_relaxed);
    for (uint64_t task = 0; task < total_tasks; ++task) {
      ExecuteGeneratedRangeBatchTask(*state, task);
    }
    return ReadyFuture();
  }

  last_worker_count.store(worker_count, std::memory_order_relaxed);
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule([state, completion, scheduled]() mutable {
        try {
          const uint64_t total_tasks =
              state->ordered_records
                  ? state->batch_count
                  : state->batch_count * state->programs.size();
          for (uint64_t task = scheduled; task < total_tasks;
               task += state->worker_count) {
            ExecuteGeneratedRangeBatchTask(*state, task);
          }
          completion.CountDown();
        } catch (const std::exception&) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained generated range worker exception"));
        } catch (...) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained unknown generated range worker "
              "exception"));
        }
      });
    } catch (const std::exception&) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: failed to schedule generated range worker"));
      break;
    } catch (...) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: unknown generated range scheduling failure"));
      break;
    }
  }
  return future;
}

template <typename SourceElement, typename ResultElement,
          typename RangeFunction>
ffi::Future ExecuteStridedGeneratedRangeProgram(
    const PreparedAffinePlan& execution, const SourceElement* source,
    ResultElement* result, ffi::ThreadPool thread_pool,
    uint64_t available_workers, uint64_t batch_count,
    RangeFunction execute_range) {
  auto context = std::make_shared<RangeFunction>(std::move(execute_range));
  const auto callback = [](const void* value, const Record& record,
                           const void* source_data, void* result_data,
                           LeafKernel leaf, uint64_t unit_start,
                           uint64_t unit_count) {
    const auto& function = *static_cast<const RangeFunction*>(value);
    function(record, static_cast<const SourceElement*>(source_data),
             static_cast<ResultElement*>(result_data), leaf, unit_start,
             unit_count);
  };
  return ExecuteStridedGeneratedErasedProgram(
      execution, source, result, sizeof(SourceElement),
      sizeof(ResultElement), thread_pool, available_workers, batch_count,
      std::static_pointer_cast<const void>(context), callback);
}

enum class OutputInitPolicy : uint8_t {
  kPlanCoverage,
  kAlreadyInitialized,
};

template <typename ResultElement>
bool InitializeRegularUnmappedOutput(const ParsedPlan& plan,
                                     ResultElement* result,
                                     uint64_t batch_count) {
  if (plan.coverage != kCoveragePartialUniqueZeroFill ||
      plan.records.size() != 1 || plan.output_size == 0) {
    return false;
  }
  const Record& record = plan.records[0];
  if (record.logical_elements == 0 || record.destination_offset < 0) {
    return false;
  }

  uint64_t row_count = 1;
  uint64_t row_elements = 1;
  uint64_t row_stride = 0;
  if (record.rank == 0) {
    row_elements = 1;
    row_stride = 1;
  } else if (record.rank == 1) {
    if (record.destination_strides[0] != 1) {
      return false;
    }
    row_elements = record.shape[0];
    row_stride = row_elements;
  } else if (record.rank == 2) {
    const std::size_t inner_axis = record.loop_axes_fastest_first[0];
    const std::size_t outer_axis = record.loop_axes_fastest_first[1];
    if (record.destination_strides[inner_axis] != 1 ||
        record.destination_strides[outer_axis] <= 0) {
      return false;
    }
    row_count = record.shape[outer_axis];
    row_elements = record.shape[inner_axis];
    row_stride =
        static_cast<uint64_t>(record.destination_strides[outer_axis]);
    if (row_stride < row_elements) {
      return false;
    }
  } else {
    return false;
  }

  const uint64_t first =
      static_cast<uint64_t>(record.destination_offset);
  uint64_t last_row_start = first;
  if (row_count > 1) {
    uint64_t row_offset = 0;
    if (!CheckedMultiply(row_count - 1, row_stride, &row_offset) ||
        !CheckedAdd(first, row_offset, &last_row_start)) {
      return false;
    }
  }
  uint64_t mapped_end = 0;
  if (!CheckedAdd(last_row_start, row_elements, &mapped_end) ||
      mapped_end > plan.output_size) {
    return false;
  }

  for (uint64_t batch = 0; batch < batch_count; ++batch) {
    ResultElement* output = result + batch * plan.output_size;
    uint64_t cursor = 0;
    uint64_t row_start = first;
    for (uint64_t row = 0; row < row_count; ++row) {
      std::fill(output + cursor, output + row_start, ResultElement{});
      cursor = row_start + row_elements;
      if (row + 1 < row_count) {
        row_start += row_stride;
      }
    }
    std::fill(output + cursor, output + plan.output_size, ResultElement{});
  }
  return true;
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename RangeFunction>
ffi::Future ExecuteGenericAffinePlan(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result,
    ffi::ThreadPool thread_pool, uint64_t expected_dtype,
    PreparedOperation expected_operation, OutputInitPolicy init_policy,
    RangeFunction execute_range,
    uint64_t expected_scalar_kind = 0,
    bool allow_source_result_alias = false,
    bool allow_batched_rank = false,
    bool coalesce_batches = true) {
  if (prepared == nullptr || prepared->execution == nullptr) {
    return ErrorFuture(Invalid("prepared state is unavailable"));
  }
  if (descriptor.size() != prepared->descriptor_bytes) {
    return ErrorFuture(Invalid("prepared descriptor byte length mismatch"));
  }
  if (prepared->operation != expected_operation) {
    return ErrorFuture(Invalid("prepared operation does not match FFI target"));
  }
  const auto execution = prepared->execution;
  const ParsedPlan& plan = execution->plan;
  if (plan.source_dtype != DescriptorDtypeCode<SourceDtype>() ||
      plan.result_dtype != DescriptorDtypeCode<ResultDtype>() ||
      plan.dtype != expected_dtype) {
    return ErrorFuture(
        Invalid("descriptor dtype does not match prepared typed target"));
  }
  if (expected_scalar_kind != 0 &&
      plan.scalar_kind != expected_scalar_kind) {
    return ErrorFuture(
        Invalid("descriptor scalar policy does not match prepared target"));
  }
  uint64_t batch_count = 0;
  ffi::Error error =
      ValidateBufferDimensions(source, result, plan.source_size,
                               plan.output_size, &batch_count,
                               allow_batched_rank);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  uint64_t source_elements = 0;
  uint64_t result_elements = 0;
  if (!CheckedMultiply(batch_count, plan.source_size, &source_elements) ||
      !CheckedMultiply(batch_count, plan.output_size, &result_elements)) {
    return ErrorFuture(Invalid("batched buffer element count overflow"));
  }
  const uint64_t source_item_size = sizeof(*source.typed_data());
  const uint64_t result_item_size = sizeof(*result->typed_data());
  error = ValidateHostElements(source_elements, source_item_size,
                               "batched source");
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  error = ValidateHostElements(result_elements, result_item_size,
                               "batched destination");
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  uint64_t source_bytes = 0;
  uint64_t result_bytes = 0;
  if (!CheckedMultiply(source_elements, source_item_size, &source_bytes) ||
      !CheckedMultiply(result_elements, result_item_size, &result_bytes)) {
    return ErrorFuture(Invalid("batched buffer byte count overflow"));
  }
  if (allow_source_result_alias) {
    if (static_cast<const void*>(source.typed_data()) !=
        static_cast<const void*>(result->typed_data())) {
      return ErrorFuture(
          Invalid("allowed source/result alias requires pointer equality"));
    }
  } else {
    error = ValidateDisjointBuffers(source.typed_data(), result->typed_data(),
                                    source_bytes, result_bytes, 1);
    if (!error.success()) {
      return ErrorFuture(std::move(error));
    }
  }

  prepared_execute_count.fetch_add(1, std::memory_order_relaxed);
  native_call_count.fetch_add(1, std::memory_order_relaxed);
  const auto* source_data = source.typed_data();
  auto* result_data = result->typed_data();
  using ResultElement = std::remove_pointer_t<decltype(result_data)>;
  if (init_policy == OutputInitPolicy::kPlanCoverage &&
      plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0 &&
      !InitializeRegularUnmappedOutput(plan, result_data, batch_count)) {
    std::fill(result_data, result_data + result_elements, ResultElement{});
  }

  const int64_t pool_threads = thread_pool.num_threads();
  const uint64_t raw_available_workers =
      pool_threads > 0 ? static_cast<uint64_t>(pool_threads) : 0;
  last_available_worker_count.store(raw_available_workers,
                                    std::memory_order_relaxed);
  uint64_t available_workers = std::min(
      raw_available_workers,
      worker_limit.load(std::memory_order_relaxed));
  uint64_t mapped_elements = 0;
  uint64_t source_work_bytes = 0;
  uint64_t result_work_bytes = 0;
  uint64_t estimated_work_bytes = 0;
  if (!CheckedMultiply(plan.copied_elements, batch_count,
                       &mapped_elements) ||
      !CheckedMultiply(mapped_elements, source_item_size,
                       &source_work_bytes) ||
      !CheckedMultiply(mapped_elements, result_item_size,
                       &result_work_bytes) ||
      !CheckedAdd(source_work_bytes, result_work_bytes,
                  &estimated_work_bytes)) {
    return ErrorFuture(Invalid("parallel work byte count overflow"));
  }
  if (init_policy == OutputInitPolicy::kPlanCoverage &&
      plan.coverage == kCoveragePartialUniqueZeroFill) {
    uint64_t zero_fill_elements = 0;
    uint64_t zero_fill_bytes = 0;
    if (!CheckedMultiply(plan.output_size, batch_count,
                         &zero_fill_elements) ||
        !CheckedMultiply(zero_fill_elements, result_item_size,
                         &zero_fill_bytes) ||
        !CheckedAdd(estimated_work_bytes, zero_fill_bytes,
                    &estimated_work_bytes)) {
      return ErrorFuture(Invalid("parallel zero-fill byte count overflow"));
    }
  }
  const bool has_copy_work =
      batch_count != 0 && plan.copied_elements != 0;
  const bool has_zero_fill_work =
      init_policy == OutputInitPolicy::kPlanCoverage &&
      plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0;
  if (execution->strided_generated_mode != 0 && has_copy_work) {
    const uint64_t generated_workers =
        estimated_work_bytes < execution->parallel_minimum_bytes
            ? UINT64_C(1)
            : available_workers;
    return ExecuteStridedGeneratedRangeProgram(
        *execution, source_data, result_data, thread_pool,
        generated_workers, batch_count, execute_range);
  }
  const bool full_compact_interval =
      !execution->force_generic && plan.records.size() == 1 &&
      plan.coverage == kCoverageCompleteUnique &&
      plan.source_size == plan.output_size &&
      plan.copied_elements == plan.source_size &&
      plan.records[0].source_offset == 0 &&
      plan.records[0].destination_offset == 0 &&
      IsCompactSameMapping(plan.records[0]);
  const auto execute_sequential = [&] {
    last_worker_count.store(
        has_copy_work || has_zero_fill_work ? 1 : 0,
        std::memory_order_relaxed);
    if (coalesce_batches && batch_count > 1 && full_compact_interval) {
      Record fused = plan.records[0];
      fused.logical_elements = source_elements;
      execute_range(fused, source_data, result_data,
                    LeafKernel::kContiguousRow, 0, source_elements);
      return ReadyFuture();
    }
    for (uint64_t batch = 0; batch < batch_count; ++batch) {
      const auto* source_batch =
          plan.source_size == 0
              ? source_data
              : source_data + batch * plan.source_size;
      auto* result_batch =
          plan.output_size == 0
              ? result_data
              : result_data + batch * plan.output_size;
      for (const RecordWork& item : execution->work) {
        const uint64_t unit_count =
            item.leaf_kernel == LeafKernel::kScalarRow
                ? plan.records[item.record_index].logical_elements
                : item.unit_count;
        execute_range(plan.records[item.record_index], source_batch,
                      result_batch, item.leaf_kernel, 0, unit_count);
      }
    }
    return ReadyFuture();
  };

  if (!has_copy_work || available_workers <= 1 ||
      estimated_work_bytes < execution->parallel_minimum_bytes) {
    return execute_sequential();
  }
  uint64_t total_chunks = 0;
  if (!CheckedMultiply(execution->chunks_per_batch, batch_count,
                       &total_chunks)) {
    return ErrorFuture(Invalid("batched parallel chunk count overflow"));
  }
  const uint64_t worker_count = std::min(available_workers, total_chunks);
  if (worker_count <= 1) {
    return execute_sequential();
  }

  using SourceElement =
      std::remove_const_t<std::remove_pointer_t<decltype(source_data)>>;
  using State = PreparedParallelState<SourceElement, ResultElement,
                                      decltype(execute_range)>;
  auto state = std::make_shared<State>(
      execution, source_data, result_data, batch_count, total_chunks,
      worker_count, std::move(execute_range));
  last_worker_count.store(worker_count, std::memory_order_relaxed);
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule([state, completion, scheduled]() mutable {
        try {
          ExecutePreparedParallelWorker(state, scheduled);
          completion.CountDown();
        } catch (const std::exception&) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained prepared worker exception"));
        } catch (...) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained unknown prepared worker exception"));
        }
      });
    } catch (const std::exception&) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: failed to schedule prepared worker"));
      break;
    } catch (...) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: unknown prepared worker scheduling failure"));
      break;
    }
  }
  return future;
}

ffi::Error ValidateSelectedScaleBufferRelationships(
    const void* base, const void* factor, void* result,
    uint64_t factor_bytes, uint64_t result_elements, uint64_t item_size,
    bool require_base_result_alias) {
  if (require_base_result_alias) {
    if (base != result) {
      return Invalid(
          "alias selected scale requires base/result pointer equality");
    }
  } else {
    ffi::Error error = ValidateDisjointBuffers(
        base, result, result_elements, result_elements, item_size);
    if (!error.success()) {
      return error;
    }
  }
  return ValidateDisjointBuffers(factor, result, factor_bytes,
                                 result_elements * item_size, 1);
}

uint64_t RuntimeScalarDtype(ffi::DataType dtype) {
  switch (dtype) {
    case ffi::F16: return kDtypeF16;
    case ffi::BF16: return kDtypeBF16;
    case ffi::F32: return kDtypeF32;
    case ffi::F64: return kDtypeF64;
    case ffi::C64: return kDtypeC64;
    case ffi::C128: return kDtypeC128;
    case ffi::PRED: return kDtypePred;
    case ffi::S8: return kDtypeS8;
    case ffi::S16: return kDtypeS16;
    case ffi::S32: return kDtypeS32;
    case ffi::S64: return kDtypeS64;
    case ffi::U8: return kDtypeU8;
    case ffi::U16: return kDtypeU16;
    case ffi::U32: return kDtypeU32;
    case ffi::U64: return kDtypeU64;
    default: return 0;
  }
}

template <ffi::DataType Dtype, bool Differential = false, typename DynamicScalarOp>
ffi::Future ExecutePreparedSelectedScale(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<Dtype> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<Dtype> result, ffi::ThreadPool thread_pool,
    uint64_t expected_dtype, DynamicScalarOp dynamic_scalar_op,
    bool require_base_result_alias = false) {
  if (prepared == nullptr || prepared->execution == nullptr) {
    return ErrorFuture(Invalid("prepared state is unavailable"));
  }
  if (descriptor.size() != prepared->descriptor_bytes) {
    return ErrorFuture(Invalid("prepared descriptor byte length mismatch"));
  }
  if (prepared->operation != PreparedOperation::kSelectedScale) {
    return ErrorFuture(Invalid("prepared operation does not match FFI target"));
  }
  const ParsedPlan& plan = prepared->execution->plan;
  if (plan.dtype != expected_dtype) {
    return ErrorFuture(
        Invalid("descriptor dtype does not match prepared typed target"));
  }
  uint64_t batch_count = 0;
  ffi::Error error =
      ValidateBufferDimensions(base, result, plan.source_size,
                               plan.output_size, &batch_count,
                               require_base_result_alias);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  const auto factor_dims = factor.dimensions();
  if (factor_dims.size() != 1 || factor_dims[0] < 0) {
    return ErrorFuture(Invalid("factor must be a rank-one physical buffer"));
  }
  const uint64_t factor_count = static_cast<uint64_t>(factor_dims[0]);
  if (factor_count != 1 && factor_count != batch_count) {
    return ErrorFuture(
        Invalid("selected scale factor count does not match batches"));
  }
  uint64_t result_elements = 0;
  if (!CheckedMultiply(batch_count, plan.output_size, &result_elements)) {
    return ErrorFuture(Invalid("selected scale element count overflow"));
  }
  const uint64_t item_size = sizeof(*result->typed_data());
  error = ValidateHostElements(result_elements, item_size, "selected base");
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  error = ValidateSelectedScaleBufferRelationships(
      base.typed_data(), factor.untyped_data(), result->typed_data(),
      factor.size_bytes(), result_elements, item_size, require_base_result_alias);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  if (require_base_result_alias) {
    alias_last_base_pointer.store(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base.typed_data())),
        std::memory_order_relaxed);
    alias_last_source_pointer.store(0, std::memory_order_relaxed);
    alias_last_factor_pointer.store(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(factor.untyped_data())),
        std::memory_order_relaxed);
    alias_last_result_pointer.store(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result->typed_data())),
        std::memory_order_relaxed);
  }
  if (!require_base_result_alias &&
      plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0) {
    std::memcpy(result->typed_data(), base.typed_data(),
                static_cast<std::size_t>(result_elements) * item_size);
  }

  const auto factor_dtype = RuntimeScalarDtype(factor.element_type());
  if (factor_dtype == 0) {
    return ErrorFuture(Invalid("unsupported scale coefficient dtype"));
  }
  const auto* factor_data = static_cast<const uint8_t*>(factor.untyped_data());
  auto* result_data = result->typed_data();
  const uint64_t output_size = plan.output_size;
  const auto factor_for_output =
      [factor_data, factor_dtype, result_data, factor_count,
       output_size](const auto* output) {
        uint64_t batch = 0;
        if (factor_count != 1 && output_size != 0) {
          batch = static_cast<uint64_t>(output - result_data) / output_size;
        }
        return ReadScalar(factor_data + (factor_count == 1 ? 0 : batch) * DtypeItemSize(factor_dtype), factor_dtype);
      };
  const auto execute_range =
      [factor_for_output, dynamic_scalar_op](const Record& record, const auto* input, auto* output,
         LeafKernel kind, uint64_t unit_start, uint64_t unit_count) {
        const auto coefficient = factor_for_output(output);
        const int coefficient_kind = Differential ? 2 : ScalarCoefficientKind(coefficient);
        BindScaleOperation<Dtype, Differential>(coefficient, record.mapped_dtype,
            coefficient_kind, dynamic_scalar_op, [&](auto scalar_op) {
              ExecuteAffineLeafRange(record, input, output, kind, unit_start,
                  unit_count, scalar_op, AssignMappedValue{});
            });
      };
  return ExecuteGenericAffinePlan<Dtype, Dtype>(
      descriptor, prepared, base, result, thread_pool, expected_dtype,
      PreparedOperation::kSelectedScale,
      OutputInitPolicy::kAlreadyInitialized, execute_range, 0,
      require_base_result_alias,
      require_base_result_alias, factor_count == 1);
}

template <typename Function>
ffi::Future ContainExceptions(Function&& function) {
  try {
    return function();
  } catch (const std::exception&) {
    return ErrorFuture(ffi::Error::Internal(
        "tensor0-stride: contained C++ standard exception"));
  } catch (...) {
    return ErrorFuture(ffi::Error::Internal(
        "tensor0-stride: contained unknown C++ exception"));
  }
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedImpl(
    ffi::Span<const uint8_t> descriptor) {
  try {
    return InstantiatePrepared(descriptor, PreparedOperation::kFreshMap);
  } catch (const std::exception&) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained affine-v8 instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown affine-v8 instantiate exception"));
  }
}

ffi::ErrorOr<std::unique_ptr<PreparedState>>
InstantiatePreparedReductionImpl(ffi::Span<const uint8_t> descriptor) {
  try {
    return InstantiatePreparedReduction(descriptor);
  } catch (const std::exception&) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained reduction instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown reduction instantiate exception"));
  }
}

ffi::ErrorOr<std::unique_ptr<PreparedState>>
InstantiatePreparedGroupedReductionImpl(
    ffi::Span<const uint8_t> descriptor) {
  try {
    return InstantiatePreparedReduction(descriptor, true);
  } catch (const std::exception&) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained grouped reduction instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown grouped reduction instantiate "
        "exception"));
  }
}

template <PreparedOperation kOperation>
ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedOperationImpl(
    ffi::Span<const uint8_t> descriptor) {
  try {
    return InstantiatePrepared(descriptor, kOperation);
  } catch (const std::exception&) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained operation instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown operation instantiate exception"));
  }
}

ffi::Future ExecutePreparedF32Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecuteGenericAffinePlan<ffi::F32, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const float* input, float* output,
           LeafKernel kind, uint64_t unit_start,
           uint64_t unit_count) {
          ExecuteForwardMapLeaf<ffi::F32, ffi::F32>(record, input, output, kind,
              unit_start, unit_count, F32Mapping{});
        });
  });
}

ffi::Future ExecutePreparedF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F16> source, ffi::ResultBuffer<ffi::F16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecuteGenericAffinePlan<ffi::F16, ffi::F16>(
        descriptor, prepared, source, result, thread_pool, kDtypeF16,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const uint16_t* input,
           uint16_t* output, LeafKernel kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteForwardMapLeaf<ffi::F16, ffi::F16>(
              record, input, output, kind, unit_start, unit_count, F16Mapping{});
        });
  });
}

ffi::Future ExecutePreparedBF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::BF16> source, ffi::ResultBuffer<ffi::BF16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecuteGenericAffinePlan<ffi::BF16, ffi::BF16>(
        descriptor, prepared, source, result, thread_pool, kDtypeBF16,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const uint16_t* input,
           uint16_t* output, LeafKernel kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteForwardMapLeaf<ffi::BF16, ffi::BF16>(record, input, output, kind,
              unit_start, unit_count, BF16Mapping{});
        });
  });
}

ffi::Future ExecutePreparedC64Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecuteGenericAffinePlan<ffi::C64, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const std::complex<float>* input,
           std::complex<float>* output, LeafKernel kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteForwardMapLeaf<ffi::C64, ffi::C64>(
              record, input, output, kind, unit_start, unit_count, C64Mapping{});
        });
  });
}

ffi::Future ExecutePreparedS32Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::S32> source, ffi::ResultBuffer<ffi::S32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecuteGenericAffinePlan<ffi::S32, ffi::S32>(
        descriptor, prepared, source, result, thread_pool, kDtypeS32,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const int32_t* input, int32_t* output,
           LeafKernel kind, uint64_t unit_start,
           uint64_t unit_count) {
          ExecuteForwardMapLeaf<ffi::S32, ffi::S32>(record, input, output, kind,
              unit_start, unit_count, IntegerMapping<int32_t>{});
        });
  });
}

#define TENSOR0_PREPARED_SCALAR_IMPL(Name, Dtype, DtypeCode, ScaleFunction)  \
  ffi::Future Name(                                                          \
      ffi::Span<const uint8_t> descriptor, PreparedState* prepared,          \
      ffi::Buffer<Dtype> source, ffi::ResultBuffer<Dtype> result,            \
      ffi::ThreadPool thread_pool) {                                         \
    return ContainExceptions([&] {                                           \
      return ExecuteGenericAffinePlan<Dtype, Dtype>(                             \
          descriptor, prepared, source, result, thread_pool, DtypeCode,      \
          PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,     \
          [](const Record& record, const auto* input, auto* output,           \
             LeafKernel kind, uint64_t unit_start,                       \
             uint64_t unit_count) {                                          \
            ExecuteForwardMapLeaf<Dtype, Dtype>(                            \
                record, input, output, kind, unit_start, unit_count,          \
                ScaleFunction{});                                           \
          });                                                                \
    });                                                                      \
  }

TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedPredImpl, ffi::PRED,
                             kDtypePred, PREDMapping)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS8Impl, ffi::S8, kDtypeS8,
                             IntegerMapping<int8_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS16Impl, ffi::S16, kDtypeS16,
                             IntegerMapping<int16_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS64Impl, ffi::S64, kDtypeS64,
                             IntegerMapping<int64_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU8Impl, ffi::U8, kDtypeU8,
                             IntegerMapping<uint8_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU16Impl, ffi::U16, kDtypeU16,
                             IntegerMapping<uint16_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU32Impl, ffi::U32, kDtypeU32,
                             IntegerMapping<uint32_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU64Impl, ffi::U64, kDtypeU64,
                             IntegerMapping<uint64_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedF64Impl, ffi::F64, kDtypeF64,
                             F64Mapping)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedC128Impl, ffi::C128, kDtypeC128,
                             C128Mapping)

#undef TENSOR0_PREPARED_SCALAR_IMPL

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename ScalarOp>
ffi::Future ExecutePreparedMixed(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    uint64_t scalar_dtype, uint64_t scalar_kind, ScalarOp scalar_op) {
  const auto execute_range =
      [scalar_op, scalar_dtype](
          const Record& record, const auto* input, auto* output,
          LeafKernel kind, uint64_t unit_start, uint64_t unit_count) {
        ExecuteForwardMapLeaf<SourceDtype, ResultDtype>(
            record, input, output, kind, unit_start, unit_count, scalar_op, scalar_dtype);
      };
  return ExecuteGenericAffinePlan<SourceDtype, ResultDtype>(
      descriptor, prepared, source, result, thread_pool, scalar_dtype,
      PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
      execute_range, scalar_kind);
}

ffi::Future ExecutePreparedF16F32ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F16> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F16, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarForwardScaleCast, F16F32Mapping{});
  });
}

ffi::Future ExecutePreparedF32F16TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::F16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::F16>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarJaxTranspose, NarrowF16Product{});
  });
}

ffi::Future ExecutePreparedF32C64ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        kScalarForwardScaleCast, F32C64Mapping{});
  });
}

ffi::Future ExecutePreparedC64F32TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C64, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        kScalarJaxTranspose, ProjectedComplexProduct<float>{});
  });
}

ffi::Future ExecutePreparedC64F32ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C64, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarForwardScaleCast, C64F32Mapping{});
  });
}

ffi::Future ExecutePreparedF32C64TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarJaxTranspose, EmbeddedRealProduct<float>{});
  });
}

ffi::Future ExecutePreparedF64C128ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F64> source, ffi::ResultBuffer<ffi::C128> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F64, ffi::C128>(
        descriptor, prepared, source, result, thread_pool, kDtypeC128,
        kScalarForwardScaleCast,
        F64C128Mapping{});
  });
}

ffi::Future ExecutePreparedC128F64TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C128> source, ffi::ResultBuffer<ffi::F64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C128, ffi::F64>(
        descriptor, prepared, source, result, thread_pool, kDtypeC128,
        kScalarJaxTranspose,
        ProjectedComplexProduct<double>{});
  });
}

ffi::Future ExecutePreparedC128F64ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C128> source, ffi::ResultBuffer<ffi::F64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C128, ffi::F64>(
        descriptor, prepared, source, result, thread_pool, kDtypeF64,
        kScalarForwardScaleCast,
        C128F64Mapping{});
  });
}

ffi::Future ExecutePreparedF64C128TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F64> source, ffi::ResultBuffer<ffi::C128> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F64, ffi::C128>(
        descriptor, prepared, source, result, thread_pool, kDtypeF64,
        kScalarJaxTranspose,
        EmbeddedRealProduct<double>{});
  });
}

ffi::Future ExecuteScaleTangentF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F16> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::F16> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::F16, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeF16, DynamicScaleF16{});
  });
}

ffi::Future ExecuteScaleTangentBF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::BF16> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::BF16> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::BF16, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeBF16, DynamicScaleBF16{});
  });
}

ffi::Future ExecuteScaleTangentF32Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::F32> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::F32, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeF32, DynamicScaleF32{});
  });
}

ffi::Future ExecuteScaleTangentF64Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F64> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::F64> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::F64, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeF64, DynamicScaleF64{});
  });
}

ffi::Future ExecuteScaleTangentC64Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::C64> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::C64, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeC64, DynamicScaleC64{});
  });
}

ffi::Future ExecuteScaleTangentC128Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C128> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::C128> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::C128, true>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeC128, DynamicScaleC128{});
  });
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype, bool Differential, typename Scale, typename Add>
ffi::Future ExecuteTypedUpdate(
    ffi::Span<const uint8_t> descriptor, ffi::Span<const uint64_t> stages,
    PreparedState* prepared, ffi::Buffer<ResultDtype> base,
    ffi::Buffer<SourceDtype> source, ffi::AnyBuffer alpha, ffi::AnyBuffer beta,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    Scale scale, Add add) {
  if (prepared == nullptr || prepared->execution == nullptr) {
    return ErrorFuture(Invalid("typed update prepared state unavailable"));
  }
  const auto& plan = prepared->execution->plan;
  if (descriptor.size() != prepared->descriptor_bytes ||
      prepared->operation != PreparedOperation::kAxpby ||
      plan.source_dtype != DescriptorDtypeCode<SourceDtype>() ||
      plan.result_dtype != DescriptorDtypeCode<ResultDtype>() ||
      plan.dtype != DescriptorDtypeCode<ResultDtype>()) {
    return ErrorFuture(Invalid("typed update descriptor or target mismatch"));
  }
  constexpr std::size_t stage_count = Differential ? 4 : 7;
  if (stages.size() != stage_count * plan.records.size()) {
    return ErrorFuture(Invalid("typed update stage count mismatch"));
  }
  if (prepared->update_types.size() != plan.records.size()) {
    return ErrorFuture(Invalid("typed update preparation missing"));
  }
  for (std::size_t index = 0; index < plan.records.size(); ++index) {
    const auto& prepared_stages = prepared->update_types[index].stages;
    if (!std::equal(prepared_stages.begin(), prepared_stages.begin() + stage_count,
                    stages.begin() + index * stage_count)) {
      return ErrorFuture(Invalid("typed update stage metadata mismatch"));
    }
  }
  const auto alpha_dtype = RuntimeScalarDtype(alpha.element_type());
  const auto beta_dtype = RuntimeScalarDtype(beta.element_type());
  if (alpha_dtype == 0 || beta_dtype == 0) {
    return ErrorFuture(Invalid("unsupported update coefficient dtype"));
  }
  uint64_t batch_count = 0;
  auto error = ValidateBufferDimensions(source, result, plan.source_size,
                                        plan.output_size, &batch_count);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  const uint64_t alpha_count = alpha.element_count();
  const uint64_t beta_count = beta.element_count();
  if ((alpha_count != 1 && alpha_count != batch_count) ||
      (beta_count != 1 && beta_count != batch_count) ||
      base.element_count() != result->element_count()) {
    return ErrorFuture(Invalid("typed update buffer shape mismatch"));
  }
  for (const auto& buffer : {std::make_pair(base.untyped_data(), base.size_bytes()),
                            std::make_pair(source.untyped_data(), source.size_bytes()),
                            std::make_pair(alpha.untyped_data(), alpha.size_bytes()),
                            std::make_pair(beta.untyped_data(), beta.size_bytes())}) {
    error = ValidateDisjointBuffers(buffer.first, result->untyped_data(),
                                    buffer.second, result->size_bytes(), 1);
    if (!error.success()) {
      return ErrorFuture(std::move(error));
    }
  }
  if (plan.coverage == kCoveragePartialUniqueZeroFill && result->size_bytes() != 0) {
    std::memcpy(result->untyped_data(), base.untyped_data(), result->size_bytes());
  }
  const auto* alpha_data = static_cast<const uint8_t*>(alpha.untyped_data());
  const auto* beta_data = static_cast<const uint8_t*>(beta.untyped_data());
  auto* output_start = result->typed_data();
  const auto* base_start = base.typed_data();
  const uint64_t output_size = plan.output_size;
  const auto execute = [=](const Record& record, const auto* input, auto* output,
                           LeafKernel kind, uint64_t unit_start, uint64_t unit_count) {
    const auto batch = output_size == 0 ? 0 : (output - output_start) / output_size;
    const auto record_index = record.semantic_index;
    const auto& types = prepared->update_types[record_index];
    const uint64_t coefficient_dtype = types.stages[0];
    uint64_t source_product_dtype = types.stages[1];
    uint64_t base_product_dtype = types.stages[2];
    uint64_t addition_dtype = types.stages[3];
    const auto raw_factor = ReadScalar(
        alpha_data + (alpha_count == 1 ? 0 : batch) * DtypeItemSize(alpha_dtype), alpha_dtype);
    const auto effective_factor = record.identity_scale
        ? ConvertScalar(raw_factor, coefficient_dtype)
        : MultiplyScalars(raw_factor, ReadMappingFactor(record), coefficient_dtype);
    const auto source_factor = effective_factor;
    const auto base_factor = ReadScalar(
        beta_data + (beta_count == 1 ? 0 : batch) * DtypeItemSize(beta_dtype), beta_dtype);
    const int source_kind = Differential ? 2 : ScalarCoefficientKind(source_factor);
    const int base_kind = Differential ? 2 : ScalarCoefficientKind(base_factor);
    if constexpr (!Differential) {
      if (source_kind == 1) source_product_dtype = DescriptorDtypeCode<SourceDtype>();
      if (base_kind == 1) base_product_dtype = DescriptorDtypeCode<ResultDtype>();
      addition_dtype = source_kind == 0 ? base_product_dtype
          : base_kind == 0 ? source_product_dtype
          : types.stages[3 + (source_kind - 1) * 2 + (base_kind - 1)];
    }
    if (!Differential && base_kind == 0) {
      BindUpdateSource<SourceDtype, ResultDtype>(source_factor, source_kind,
          source_product_dtype, scale, [&](auto scalar_op) {
            ExecuteAffineLeafRange(record, input, output, kind, unit_start,
                unit_count, scalar_op, AssignMappedValue{});
          });
      return;
    }
    const auto result_dtype = DescriptorDtypeCode<ResultDtype>();
    if ((source_kind == 0 || source_product_dtype == result_dtype) &&
        base_product_dtype == result_dtype && addition_dtype == result_dtype) {
      auto first = ConvertScalar<ResultDtype>(source_factor);
      auto second = ConvertScalar<ResultDtype>(base_factor);
      const auto* original = base_start + (output - output_start);
      if constexpr (!Differential) {
        BindUpdateCombination(original, output, first, second, source_kind,
            base_kind, scale, add, ScalarCast<SourceDtype, ResultDtype>{},
            [&](auto scalar_op, auto write_op) {
              ExecuteAffineLeafRange(record, input, output, kind, unit_start,
                  unit_count, scalar_op, write_op);
            });
      } else {
        ExecuteAffineLeafRange(record, input, output, kind, unit_start, unit_count,
            BoundScalarScale<decltype(first), Scale>{first, scale},
            CombineOriginalValue<true, 2, ffi::NativeType<ResultDtype>, decltype(second), Scale, Add>{
                original, output, second, scale, add});
      }
      return;
    }
    if (BindNativeUpdateCombination<SourceDtype, ResultDtype>(
            base_start, output_start, source_factor, base_factor, source_kind, base_kind,
            source_product_dtype, base_product_dtype, addition_dtype,
            [&](auto scalar_op, auto write_op) {
              ExecuteAffineLeafRange(record, input, output, kind, unit_start,
                  unit_count, scalar_op, write_op);
            })) {
      return;
    }
    ExecuteAffineLeafRange(record, input, output, kind, unit_start, unit_count,
        TypedUpdateTerm<SourceDtype>{source_factor, source_product_dtype, source_kind},
        CombineTypedOriginalValue<ResultDtype>{base_start, output_start, base_factor,
            base_product_dtype, addition_dtype, source_kind, base_kind});
  };
  return ExecuteGenericAffinePlan<SourceDtype, ResultDtype>(
      descriptor, prepared, source, result, thread_pool,
      DescriptorDtypeCode<ResultDtype>(), PreparedOperation::kAxpby,
      OutputInitPolicy::kAlreadyInitialized, execute, kScalarForwardScaleCast, false, false,
      alpha_count == 1 && beta_count == 1);
}

#define TENSOR0_AXPBY_IMPL(Name, Dtype, DtypeCode, Scale, Add)                \
  ffi::Future Name(                                                          \
      ffi::Span<const uint8_t> descriptor, ffi::Span<const uint64_t> stages, PreparedState* prepared,          \
      ffi::Buffer<Dtype> base, ffi::Buffer<Dtype> source,                    \
      ffi::AnyBuffer alpha, ffi::AnyBuffer beta,                     \
      ffi::ResultBuffer<Dtype> result, ffi::ThreadPool thread_pool) {        \
    return ContainExceptions([&] {                                           \
      return ExecuteTypedUpdate<Dtype, Dtype, true>(                                    \
          descriptor, stages, prepared, base, source, alpha, beta, result,          \
          thread_pool, Scale{}, Add{});                           \
    });                                                                      \
  }

TENSOR0_AXPBY_IMPL(ExecuteAxpbyF16Impl, ffi::F16, kDtypeF16,
                    DynamicScaleF16, AddF16)
TENSOR0_AXPBY_IMPL(ExecuteAxpbyBF16Impl, ffi::BF16, kDtypeBF16,
                    DynamicScaleBF16, AddBF16)
TENSOR0_AXPBY_IMPL(ExecuteAxpbyF32Impl, ffi::F32, kDtypeF32,
                    DynamicScaleF32, AddF32)
TENSOR0_AXPBY_IMPL(ExecuteAxpbyF64Impl, ffi::F64, kDtypeF64,
                    DynamicScaleF64, AddF64)
TENSOR0_AXPBY_IMPL(ExecuteAxpbyC64Impl, ffi::C64, kDtypeC64,
                    DynamicScaleC64, AddC64)
TENSOR0_AXPBY_IMPL(ExecuteAxpbyC128Impl, ffi::C128, kDtypeC128,
                    DynamicScaleC128, AddC128)

#undef TENSOR0_AXPBY_IMPL

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateTypedUpdate(
    ffi::Span<const uint8_t> descriptor, ffi::Span<const uint64_t> stages) {
  auto state = InstantiatePreparedOperationImpl<PreparedOperation::kAxpby>(descriptor);
  if (state.has_error()) {
    return state;
  }
  const auto record_count = (*state)->execution->plan.records.size();
  if (stages.size() != record_count * 4 && stages.size() != record_count * 7) {
    return ffi::Unexpected(Invalid("typed update stage count mismatch"));
  }
  auto& prepared = **state;
  const auto& plan = prepared.execution->plan;
  const std::size_t stage_count = stages.size() == record_count * 7 ? 7 : 4;
  for (std::size_t index = 0; index < plan.records.size(); ++index) {
    UpdateRecordTypes types;
    for (std::size_t stage = 0; stage < stage_count; ++stage) {
      const auto dtype = stages[index * stage_count + stage];
      if (dtype < kDtypeF32 || dtype > kDtypeU64) {
        return ffi::Unexpected(Invalid("unsupported update stage dtype"));
      }
      types.stages[stage] = dtype;
    }
    prepared.update_types[index] = types;
  }
  return state;
}

#define TENSOR0_UPDATE_IMPL(Name, Source, Dtype, Scale, Add)       \
  ffi::Future Name(                                                          \
      ffi::Span<const uint8_t> descriptor, ffi::Span<const uint64_t> stages, PreparedState* prepared,          \
      ffi::Buffer<Dtype> base, ffi::Buffer<Source> source,                   \
      ffi::AnyBuffer alpha, ffi::AnyBuffer beta,                     \
      ffi::ResultBuffer<Dtype> result, ffi::ThreadPool thread_pool) {        \
    return ContainExceptions([&] {                                          \
      return ExecuteTypedUpdate<Source, Dtype, false>(                    \
          descriptor, stages, prepared, base, source, alpha, beta, result,          \
          thread_pool, Scale{}, Add{});                         \
    });                                                                     \
  }

TENSOR0_UPDATE_IMPL(ExecuteUpdateF32Impl, ffi::F32, ffi::F32,
                    DynamicScaleF32, AddF32)
TENSOR0_UPDATE_IMPL(ExecuteUpdateF16Impl, ffi::F16, ffi::F16,
                    DynamicScaleF16, AddF16)
TENSOR0_UPDATE_IMPL(ExecuteUpdateBF16Impl, ffi::BF16, ffi::BF16,
                    DynamicScaleBF16, AddBF16)
TENSOR0_UPDATE_IMPL(ExecuteUpdateC64Impl, ffi::C64, ffi::C64,
                    DynamicScaleC64, AddC64)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS32Impl, ffi::S32, ffi::S32,
                    DynamicScaleInteger<int32_t>, AddIntegerOp<int32_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdatePredImpl, ffi::PRED, ffi::PRED,
                    DynamicScalePred, AddPred)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS8Impl, ffi::S8, ffi::S8,
                    DynamicScaleInteger<int8_t>, AddIntegerOp<int8_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS16Impl, ffi::S16, ffi::S16,
                    DynamicScaleInteger<int16_t>, AddIntegerOp<int16_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS64Impl, ffi::S64, ffi::S64,
                    DynamicScaleInteger<int64_t>, AddIntegerOp<int64_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateU8Impl, ffi::U8, ffi::U8,
                    DynamicScaleInteger<uint8_t>, AddIntegerOp<uint8_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateU16Impl, ffi::U16, ffi::U16,
                    DynamicScaleInteger<uint16_t>, AddIntegerOp<uint16_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateU32Impl, ffi::U32, ffi::U32,
                    DynamicScaleInteger<uint32_t>, AddIntegerOp<uint32_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateU64Impl, ffi::U64, ffi::U64,
                    DynamicScaleInteger<uint64_t>, AddIntegerOp<uint64_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateF64Impl, ffi::F64, ffi::F64,
                    DynamicScaleF64, AddF64)
TENSOR0_UPDATE_IMPL(ExecuteUpdateC128Impl, ffi::C128, ffi::C128,
                    DynamicScaleC128, AddC128)
TENSOR0_UPDATE_IMPL(ExecuteUpdateF16F32Impl, ffi::F16, ffi::F32,
                    DynamicScaleF32, AddF32)
TENSOR0_UPDATE_IMPL(ExecuteUpdateF32C64Impl, ffi::F32, ffi::C64,
                    DynamicScaleC64, AddC64)
TENSOR0_UPDATE_IMPL(ExecuteUpdateC64F32Impl, ffi::C64, ffi::F32,
                    DynamicScaleF32, AddF32)
TENSOR0_UPDATE_IMPL(ExecuteUpdateF64C128Impl, ffi::F64, ffi::C128,
                    DynamicScaleC128, AddC128)
TENSOR0_UPDATE_IMPL(ExecuteUpdateC128F64Impl, ffi::C128, ffi::F64,
                    DynamicScaleF64, AddF64)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS32PredImpl, ffi::S32, ffi::PRED,
                    DynamicScalePred, AddPred)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS32S8Impl, ffi::S32, ffi::S8,
                    DynamicScaleInteger<int8_t>, AddIntegerOp<int8_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateS32S16Impl, ffi::S32, ffi::S16,
                    DynamicScaleInteger<int16_t>, AddIntegerOp<int16_t>)
TENSOR0_UPDATE_IMPL(ExecuteUpdateU32U8Impl, ffi::U32, ffi::U8,
                    DynamicScaleInteger<uint8_t>, AddIntegerOp<uint8_t>)

#undef TENSOR0_UPDATE_IMPL

ffi::Future ExecuteSelectedScaleAliasF32Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::F32> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::F32>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeF32, DynamicScaleF32{}, true);
  });
}

ffi::Future ExecuteSelectedScaleAliasC64Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> base, ffi::AnyBuffer factor,
    ffi::ResultBuffer<ffi::C64> result, ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedSelectedScale<ffi::C64>(
        descriptor, prepared, base, factor, result, thread_pool,
        kDtypeC64, DynamicScaleC64{}, true);
  });
}

// Prepared-state lifecycle, affine scheduling, dtype policies, and update execution.
