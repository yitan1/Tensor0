struct SemanticReductionRecord {
  std::size_t rank = 0;
  int64_t input_offset = 0;
  int64_t output_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
  bool identity_scale = true;
  uint64_t mapped_dtype = 0;
  uint64_t logical_elements = 0;
  std::vector<uint64_t> shape;
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_strides;
  std::vector<bool> reduction_axes;
};

struct ReductionRecord {
  std::size_t rank = 0;
  int64_t input_offset = 0;
  int64_t output_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
  bool identity_scale = true;
  uint64_t mapped_dtype = 0;
  uint64_t map_axis_mask = 0;
  uint64_t reduction_axis_mask = 0;
  uint64_t output_count = 0;
  uint64_t reduction_count = 0;
  std::array<uint64_t, kMaximumRank> shape{};
  std::array<int64_t, kMaximumRank> input_strides{};
  std::array<int64_t, kMaximumRank> output_strides{};
  std::array<std::size_t, kMaximumRank> input_axes_fastest_first{};
  std::array<std::size_t, kMaximumRank> map_loop_order{};
  std::array<std::size_t, kMaximumRank> reduction_loop_order{};
  std::size_t map_rank = 0;
  std::size_t reduction_rank = 0;
};

struct ReductionGroup {
  std::vector<std::size_t> record_indices;
  int64_t output_minimum = 0;
  int64_t output_maximum = 0;
  uint64_t output_count = 0;
  uint64_t total_reduction_count = 0;
  uint64_t outputs_per_chunk = 0;
  uint64_t chunk_begin = 0;
  uint64_t chunk_count = 0;
};

struct GeneratedReductionWork {
  std::size_t owner_index = 0;
  uint64_t output_start = 0;
  uint64_t output_count = 0;
};

struct ParsedReductionPlan {
  uint64_t input_size = 0;
  uint64_t output_size = 0;
  uint64_t input_dtype = 0;
  uint64_t mapped_dtype = 0;
  uint64_t output_dtype = 0;
  uint64_t scalar_policy = 0;
  uint64_t preferred_chunk_bytes = 0;
  uint64_t parallel_minimum_bytes = 0;
  std::vector<ReductionRecord> records;
  bool grouped_output_owner = false;
  uint64_t grouped_chunks_per_batch = 0;
  uint64_t grouped_work_bytes_per_batch = 0;
  std::vector<ReductionGroup> groups;
  uint64_t strided_generated_mode = 0;
  bool generated_ordered_records = false;
  std::vector<GeneratedReductionWork> generated_work;
};

struct ParsedDotPlan {
  uint64_t left_size = 0;
  uint64_t right_size = 0;
  uint64_t dtype = 0;
  bool conjugate_left = false;
  std::vector<Record> records;
};

uint64_t TrackedDotPlanBytes(const ParsedDotPlan& plan) {
  return sizeof(ParsedDotPlan) + plan.records.capacity() * sizeof(Record);
}

ffi::Error ParseDotDescriptor(ffi::Span<const uint8_t> descriptor,
                              ParsedDotPlan* plan) {
  const DescriptorReader reader(descriptor);
  if (!reader.HasWholeWords()) {
    return Invalid("dot descriptor byte length is not divisible by eight");
  }
  if (reader.size() < kDotDescriptorHeaderWords) {
    return Invalid("dot descriptor is truncated");
  }
  std::array<uint64_t, kDotDescriptorHeaderWords> header{};
  for (std::size_t index = 0; index < header.size(); ++index) {
    if (!reader.Read(index, &header[index])) {
      return Invalid("dot descriptor header is truncated");
    }
  }
  if (header[0] != kDotDescriptorMagic) {
    return Invalid("dot descriptor magic mismatch");
  }
  if (header[1] != kDotDescriptorVersion) {
    return Invalid("dot descriptor version mismatch");
  }
  if (header[2] != reader.size()) {
    return Invalid("dot descriptor word count mismatch");
  }
  if (header[6] != kDtypeF32 && header[6] != kDtypeC64) {
    return Invalid("dot descriptor dtype is unsupported");
  }
  if (header[7] > 1) {
    return Invalid("dot descriptor conjugation flag is invalid");
  }
  if (header[5] > reader.size()) {
    return Invalid("dot descriptor record count exceeds descriptor size");
  }

  ParsedDotPlan parsed;
  parsed.left_size = header[3];
  parsed.right_size = header[4];
  parsed.dtype = header[6];
  parsed.conjugate_left = header[7] != 0;
  const uint64_t item_size = DtypeItemSize(parsed.dtype);
  ffi::Error error =
      ValidateHostElements(parsed.left_size, item_size, "dot left input");
  if (!error.success()) {
    return error;
  }
  error = ValidateHostElements(parsed.right_size, item_size,
                               "dot right input");
  if (!error.success()) {
    return error;
  }

  std::size_t cursor = kDotDescriptorHeaderWords;
  parsed.records.reserve(static_cast<std::size_t>(header[5]));
  for (uint64_t record_index = 0; record_index < header[5]; ++record_index) {
    std::array<uint64_t, 4> fixed{};
    for (uint64_t& value : fixed) {
      if (!reader.Read(cursor++, &value)) {
        return Invalid("dot record header is truncated");
      }
    }
    if (fixed[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        fixed[2] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return Invalid("dot record offset exceeds int64");
    }
    const std::size_t rank = static_cast<std::size_t>(fixed[0]);
    uint64_t record_words = 0;
    if (!CheckedMultiply(fixed[0], 3, &record_words) ||
        record_words > reader.size() - cursor) {
      return Invalid("dot record is truncated");
    }
    SemanticRecord semantic;
    semantic.rank = rank;
    semantic.source_offset = static_cast<int64_t>(fixed[1]);
    semantic.destination_offset = static_cast<int64_t>(fixed[2]);
    semantic.logical_elements = fixed[3];
    semantic.shape.resize(rank);
    semantic.source_strides.resize(rank);
    semantic.destination_strides.resize(rank);
    uint64_t calculated_elements = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (!reader.Read(cursor++, &semantic.shape[axis]) ||
          !CheckedMultiply(calculated_elements, semantic.shape[axis],
                           &calculated_elements)) {
        return Invalid("dot logical shape is invalid");
      }
    }
    if (calculated_elements != semantic.logical_elements) {
      return Invalid("dot logical element count mismatch");
    }
    for (std::size_t axis = 0; axis < rank; ++axis) {
      uint64_t word = 0;
      if (!reader.Read(cursor++, &word)) {
        return Invalid("dot left strides are truncated");
      }
      semantic.source_strides[axis] = DecodeStrideWord(word);
    }
    for (std::size_t axis = 0; axis < rank; ++axis) {
      uint64_t word = 0;
      if (!reader.Read(cursor++, &word)) {
        return Invalid("dot right strides are truncated");
      }
      semantic.destination_strides[axis] = DecodeStrideWord(word);
    }
    int64_t left_minimum = 0;
    int64_t left_maximum = 0;
    int64_t right_minimum = 0;
    int64_t right_maximum = 0;
    error = AddressBounds(semantic, true, &left_minimum, &left_maximum);
    if (!error.success()) {
      return error;
    }
    error = AddressBounds(semantic, false, &right_minimum, &right_maximum);
    if (!error.success()) {
      return error;
    }
    if (semantic.logical_elements == 0) {
      if (static_cast<uint64_t>(semantic.source_offset) > parsed.left_size ||
          static_cast<uint64_t>(semantic.destination_offset) >
              parsed.right_size) {
        return Invalid("empty dot record offset exceeds input size");
      }
    } else if (left_minimum < 0 || right_minimum < 0 ||
               static_cast<uint64_t>(left_maximum) >= parsed.left_size ||
               static_cast<uint64_t>(right_maximum) >= parsed.right_size) {
      return Invalid("dot record address exceeds input size");
    }
    OptimizeSemanticRecordForExecution(&semantic);
    Record record;
    error = MaterializePreparedRecord(semantic, &record);
    if (!error.success()) {
      return error;
    }
    parsed.records.push_back(record);
  }
  if (cursor != reader.size()) {
    return Invalid("dot descriptor has trailing words");
  }
  *plan = std::move(parsed);
  return ffi::Error::Success();
}

inline constexpr uint64_t kReductionPreferredChunkBytes = 256 * 1024;
inline constexpr uint64_t kReductionParallelMinimumBytes = 256 * 1024;
inline constexpr uint64_t kGroupedMinimumWorkBytesPerChunk = 32 * 1024;

inline constexpr uint64_t kReductionFiberParallelUseProductionMode =
    std::numeric_limits<uint64_t>::max();
inline constexpr uint64_t kReductionFiberParallelDisabled = 0;
inline constexpr uint64_t kReductionFiberParallelForced = 1;
inline constexpr uint64_t kReductionFiberMinimumChunkBytes = 64 * 1024;
inline constexpr uint64_t kReductionFiberMaximumChunks = 16;
inline constexpr uint64_t kReductionFiberMaximumScratchBytes = 16 * 1024 * 1024;

uint64_t TrackedReductionPlanBytes(const ParsedReductionPlan& plan) {
  uint64_t tracked = sizeof(ParsedReductionPlan) +
                     plan.records.capacity() * sizeof(ReductionRecord) +
                     plan.groups.capacity() * sizeof(ReductionGroup) +
                     plan.generated_work.capacity() *
                         sizeof(GeneratedReductionWork);
  for (const ReductionGroup& group : plan.groups) {
    tracked += group.record_indices.capacity() * sizeof(std::size_t);
  }
  return tracked;
}

ffi::Error ValidateReductionAxisOrder(
    const std::array<std::size_t, kMaximumRank>& order,
    std::size_t count, uint64_t expected_mask, std::size_t rank,
    const char* field) {
  uint64_t actual_mask = 0;
  for (std::size_t position = 0; position < count; ++position) {
    const std::size_t axis = order[position];
    if (axis >= rank || (actual_mask & (UINT64_C(1) << axis)) != 0) {
      return Invalid(std::string(field) + " is not a unique axis order");
    }
    actual_mask |= UINT64_C(1) << axis;
  }
  if (actual_mask != expected_mask) {
    return Invalid(std::string(field) + " does not match its axis mask");
  }
  return ffi::Error::Success();
}

ffi::Error ValidateReductionOutputMapInjective(
    const ReductionRecord& record, uint64_t axis_mask) {
  const auto& strides = record.output_strides;
  std::array<std::size_t, kMaximumRank> axes{};
  std::size_t count = 0;
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    if ((axis_mask & (UINT64_C(1) << axis)) != 0) {
      axes[count++] = axis;
    }
  }
  std::stable_sort(
      axes.begin(), axes.begin() + count,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(strides[left]), left) <
               std::make_pair(AbsoluteStride(strides[right]), right);
      });
  uint64_t covered_span = 0;
  for (std::size_t position = 0; position < count; ++position) {
    const std::size_t axis = axes[position];
    if (record.shape[axis] <= 1) {
      continue;
    }
    const uint64_t stride = AbsoluteStride(strides[axis]);
    if (stride <= covered_span) {
      return Invalid(
          "reduction output map does not prove an injective view");
    }
    uint64_t span = 0;
    if (!CheckedMultiply(record.shape[axis] - 1, stride, &span) ||
        !CheckedAdd(covered_span, span, &covered_span)) {
      return Invalid("reduction injectivity arithmetic overflow");
    }
  }
  return ffi::Error::Success();
}

ffi::Error ReductionAddressBounds(const ReductionRecord& record, bool input,
                                  uint64_t axis_mask, int64_t* minimum,
                                  int64_t* maximum) {
  const auto& strides = input ? record.input_strides : record.output_strides;
  int64_t low = input ? record.input_offset : record.output_offset;
  int64_t high = low;
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    if ((axis_mask & (UINT64_C(1) << axis)) == 0 ||
        record.shape[axis] <= 1) {
      continue;
    }
    uint64_t magnitude = 0;
    if (!CheckedMultiply(record.shape[axis] - 1,
                         AbsoluteStride(strides[axis]), &magnitude) ||
        magnitude >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return Invalid("reduction address arithmetic overflow");
    }
    if (strides[axis] < 0) {
      if (magnitude > static_cast<uint64_t>(low)) {
        return Invalid("reduction address precedes storage");
      }
      low -= static_cast<int64_t>(magnitude);
    } else {
      if (magnitude > static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max() - high)) {
        return Invalid("reduction address arithmetic overflow");
      }
      high += static_cast<int64_t>(magnitude);
    }
  }
  *minimum = low;
  *maximum = high;
  return ffi::Error::Success();
}

ffi::Error ValidateReductionRecord(ReductionRecord* record,
                                   uint64_t input_size,
                                   uint64_t output_size) {
  const uint64_t valid_mask =
      record->rank == 0 ? 0 : (UINT64_C(1) << record->rank) - 1;
  if ((record->map_axis_mask & ~valid_mask) != 0 ||
      (record->reduction_axis_mask & ~valid_mask) != 0 ||
      (record->map_axis_mask & record->reduction_axis_mask) != 0 ||
      (record->map_axis_mask | record->reduction_axis_mask) != valid_mask) {
    return Invalid("reduction map/reduction axes do not partition rank");
  }
  uint64_t output_count = 1;
  uint64_t reduction_count = 1;
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    const uint64_t extent = record->shape[axis];
    const bool reduction_axis =
        (record->reduction_axis_mask & (UINT64_C(1) << axis)) != 0;
    if (reduction_axis && record->output_strides[axis] != 0) {
      return Invalid("reduction output depends on a reduction axis");
    }
    if (!reduction_axis && extent > 1 &&
        record->output_strides[axis] == 0) {
      return Invalid("reduction output map has a zero nontrivial stride");
    }
    uint64_t* count =
        (record->map_axis_mask & (UINT64_C(1) << axis)) != 0
            ? &output_count
            : &reduction_count;
    if (!CheckedMultiply(*count, extent, count)) {
      return Invalid("reduction fiber element count overflow");
    }
  }
  if (output_count != record->output_count ||
      reduction_count != record->reduction_count) {
    return Invalid("reduction encoded fiber counts mismatch");
  }
  ffi::Error error =
      ValidateReductionOutputMapInjective(*record, record->map_axis_mask);
  if (!error.success()) {
    return error;
  }

  std::array<std::size_t, kMaximumRank> expected_input{};
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    expected_input[axis] = axis;
  }
  std::stable_sort(
      expected_input.begin(), expected_input.begin() + record->rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(record->input_strides[left]),
                              left) <
               std::make_pair(AbsoluteStride(record->input_strides[right]),
                              right);
      });
  if (!std::equal(expected_input.begin(),
                  expected_input.begin() + record->rank,
                  record->input_axes_fastest_first.begin())) {
    return Invalid("reduction compiled input axis order mismatch");
  }
  std::array<std::size_t, kMaximumRank> expected_map{};
  std::array<std::size_t, kMaximumRank> expected_reduction{};
  std::size_t map_rank = 0;
  std::size_t reduction_rank = 0;
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    if ((record->map_axis_mask & (UINT64_C(1) << axis)) != 0) {
      expected_map[map_rank++] = axis;
    } else {
      expected_reduction[reduction_rank++] = axis;
    }
  }
  std::stable_sort(
      expected_map.begin(), expected_map.begin() + map_rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_tuple(AbsoluteStride(record->output_strides[left]),
                               AbsoluteStride(record->input_strides[left]),
                               left) <
               std::make_tuple(AbsoluteStride(record->output_strides[right]),
                               AbsoluteStride(record->input_strides[right]),
                               right);
      });
  std::stable_sort(
      expected_reduction.begin(),
      expected_reduction.begin() + reduction_rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(record->input_strides[left]),
                              left) <
               std::make_pair(AbsoluteStride(record->input_strides[right]),
                              right);
      });
  if (record->map_rank != map_rank ||
      record->reduction_rank != reduction_rank ||
      !std::equal(expected_map.begin(), expected_map.begin() + map_rank,
                  record->map_loop_order.begin()) ||
      !std::equal(expected_reduction.begin(),
                  expected_reduction.begin() + reduction_rank,
                  record->reduction_loop_order.begin())) {
    return Invalid("reduction compiled loop order mismatch");
  }

  const bool logical_empty = output_count == 0 || reduction_count == 0;
  if (logical_empty) {
    if (static_cast<uint64_t>(record->input_offset) > input_size) {
      return Invalid("empty reduction input offset exceeds storage");
    }
  } else {
    int64_t input_minimum = 0;
    int64_t input_maximum = 0;
    error = ReductionAddressBounds(*record, true, valid_mask,
                                   &input_minimum, &input_maximum);
    if (!error.success()) {
      return error;
    }
    if (input_minimum < 0 ||
        static_cast<uint64_t>(input_maximum) >= input_size) {
      return Invalid("reduction input exceeds storage");
    }
  }
  if (output_count == 0) {
    if (static_cast<uint64_t>(record->output_offset) > output_size) {
      return Invalid("empty reduction output offset exceeds storage");
    }
  } else {
    int64_t output_minimum = 0;
    int64_t output_maximum = 0;
    error = ReductionAddressBounds(*record, false, record->map_axis_mask,
                                   &output_minimum, &output_maximum);
    if (!error.success()) {
      return error;
    }
    if (output_minimum < 0 ||
        static_cast<uint64_t>(output_maximum) >= output_size) {
      return Invalid("reduction output map exceeds storage");
    }
  }
  return ffi::Error::Success();
}

void OptimizeSemanticReductionRecordForExecution(
    SemanticReductionRecord* record) {
  if (record->logical_elements == 0 || record->rank == 0) {
    return;
  }
  std::vector<uint64_t> shape;
  std::vector<int64_t> input_strides;
  std::vector<int64_t> output_strides;
  std::vector<bool> reduction_axes;
  shape.reserve(record->rank);
  input_strides.reserve(record->rank);
  output_strides.reserve(record->rank);
  reduction_axes.reserve(record->rank);
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    if (record->shape[axis] == 1 && !record->reduction_axes[axis]) {
      continue;
    }
    shape.push_back(record->shape[axis]);
    input_strides.push_back(record->input_strides[axis]);
    output_strides.push_back(record->output_strides[axis]);
    reduction_axes.push_back(record->reduction_axes[axis]);
  }
  record->shape = std::move(shape);
  record->input_strides = std::move(input_strides);
  record->output_strides = std::move(output_strides);
  record->reduction_axes = std::move(reduction_axes);
  record->rank = record->shape.size();

  std::size_t axis = 0;
  while (axis + 1 < record->rank) {
    int64_t left_input_span = 0;
    int64_t left_output_span = 0;
    int64_t right_input_span = 0;
    int64_t right_output_span = 0;
    const bool left_fastest =
        CheckedMultiplyStride(record->input_strides[axis],
                              record->shape[axis], &left_input_span) &&
        CheckedMultiplyStride(record->output_strides[axis],
                              record->shape[axis], &left_output_span) &&
        left_input_span == record->input_strides[axis + 1] &&
        left_output_span == record->output_strides[axis + 1];
    const bool right_fastest =
        CheckedMultiplyStride(record->input_strides[axis + 1],
                              record->shape[axis + 1],
                              &right_input_span) &&
        CheckedMultiplyStride(record->output_strides[axis + 1],
                              record->shape[axis + 1],
                              &right_output_span) &&
        right_input_span == record->input_strides[axis] &&
        right_output_span == record->output_strides[axis];
    if ((!left_fastest && !right_fastest) ||
        record->reduction_axes[axis] != record->reduction_axes[axis + 1]) {
      ++axis;
      continue;
    }
    uint64_t merged_shape = 0;
    if (!CheckedMultiply(record->shape[axis], record->shape[axis + 1],
                         &merged_shape)) {
      ++axis;
      continue;
    }
    record->shape[axis] = merged_shape;
    if (right_fastest) {
      record->input_strides[axis] = record->input_strides[axis + 1];
      record->output_strides[axis] = record->output_strides[axis + 1];
    }
    record->shape.erase(record->shape.begin() + axis + 1);
    record->input_strides.erase(record->input_strides.begin() + axis + 1);
    record->output_strides.erase(record->output_strides.begin() + axis + 1);
    record->reduction_axes.erase(record->reduction_axes.begin() + axis + 1);
    --record->rank;
    if (axis != 0) {
      --axis;
    }
  }
}

ffi::Error MaterializeReductionRecord(const SemanticReductionRecord& semantic,
                                      uint64_t input_size,
                                      uint64_t output_size,
                                      ReductionRecord* record) {
  if (semantic.rank > kMaximumRank) {
    return Invalid("effective reduction rank exceeds native limit " +
                   std::to_string(kMaximumRank));
  }
  record->rank = semantic.rank;
  record->input_offset = semantic.input_offset;
  record->output_offset = semantic.output_offset;
  record->scale_real_bits = semantic.scale_real_bits;
  record->scale_imaginary_bits = semantic.scale_imaginary_bits;
  record->identity_scale = semantic.identity_scale;
  record->mapped_dtype = semantic.mapped_dtype;
  record->output_count = 1;
  record->reduction_count = 1;
  for (std::size_t axis = 0; axis < semantic.rank; ++axis) {
    record->shape[axis] = semantic.shape[axis];
    record->input_strides[axis] = semantic.input_strides[axis];
    record->output_strides[axis] = semantic.output_strides[axis];
    record->input_axes_fastest_first[axis] = axis;
    if (semantic.reduction_axes[axis]) {
      record->reduction_axis_mask |= UINT64_C(1) << axis;
      record->reduction_loop_order[record->reduction_rank++] = axis;
      if (!CheckedMultiply(record->reduction_count, semantic.shape[axis],
                           &record->reduction_count)) {
        return Invalid("reduction fiber element count overflow");
      }
    } else {
      record->map_axis_mask |= UINT64_C(1) << axis;
      record->map_loop_order[record->map_rank++] = axis;
      if (!CheckedMultiply(record->output_count, semantic.shape[axis],
                           &record->output_count)) {
        return Invalid("reduction output element count overflow");
      }
    }
  }
  std::stable_sort(
      record->input_axes_fastest_first.begin(),
      record->input_axes_fastest_first.begin() + record->rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(record->input_strides[left]), left) <
               std::make_pair(AbsoluteStride(record->input_strides[right]), right);
      });
  std::stable_sort(
      record->map_loop_order.begin(),
      record->map_loop_order.begin() + record->map_rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_tuple(AbsoluteStride(record->output_strides[left]),
                               AbsoluteStride(record->input_strides[left]), left) <
               std::make_tuple(AbsoluteStride(record->output_strides[right]),
                               AbsoluteStride(record->input_strides[right]), right);
      });
  std::stable_sort(
      record->reduction_loop_order.begin(),
      record->reduction_loop_order.begin() + record->reduction_rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(record->input_strides[left]), left) <
               std::make_pair(AbsoluteStride(record->input_strides[right]), right);
      });
  return ValidateReductionRecord(record, input_size, output_size);
}

ffi::Error ParseReductionDescriptor(ffi::Span<const uint8_t> descriptor,
                                    ParsedReductionPlan* plan) {
  const DescriptorReader reader(descriptor);
  if (!reader.HasWholeWords()) {
    return Invalid("reduction descriptor byte length is not divisible by eight");
  }
  if (reader.size() < kReductionDescriptorHeaderWords) {
    return Invalid("reduction descriptor is truncated");
  }
  std::array<uint64_t, kReductionDescriptorHeaderWords> header{};
  for (std::size_t index = 0; index < header.size(); ++index) {
    if (!reader.Read(index, &header[index])) {
      return Invalid("reduction descriptor header is truncated");
    }
  }
  if (header[0] != kReductionDescriptorMagic ||
      header[1] != kReductionDescriptorVersion ||
      header[2] != reader.size()) {
    return Invalid("reduction descriptor header mismatch");
  }
  plan->input_size = header[3];
  plan->output_size = header[4];
  const uint64_t record_count_word = header[5];
  plan->input_dtype = header[6];
  plan->output_dtype = header[7];
  plan->scalar_policy = header[8];
  plan->mapped_dtype =
      plan->scalar_policy == kReductionScalarForwardScaleCast
          ? plan->output_dtype
          : plan->input_dtype;
  plan->preferred_chunk_bytes = kReductionPreferredChunkBytes;
  plan->parallel_minimum_bytes = kReductionParallelMinimumBytes;
  if (record_count_word > reader.size() ||
      record_count_word > std::numeric_limits<std::size_t>::max()) {
    return Invalid("reduction record count exceeds descriptor size");
  }
  const bool forward_scale_cast =
      plan->scalar_policy == kReductionScalarForwardScaleCast;
  const bool jax_transpose =
      plan->scalar_policy == kReductionScalarJaxTranspose;
  if ((!forward_scale_cast && !jax_transpose) ||
      !IsReductionDtype(plan->input_dtype) ||
      !IsReductionDtype(plan->output_dtype)) {
    return Invalid("reduction descriptor dtype/scalar policy is unsupported");
  }
  ffi::Error error = ValidateHostElements(
      plan->input_size, DtypeItemSize(plan->input_dtype), "reduction input");
  if (!error.success()) {
    return error;
  }
  error = ValidateHostElements(plan->output_size,
                               DtypeItemSize(plan->output_dtype),
                               "reduction output");
  if (!error.success()) {
    return error;
  }

  const std::size_t record_count =
      static_cast<std::size_t>(record_count_word);
  plan->records.clear();
  plan->records.reserve(record_count);
  std::size_t cursor = kReductionDescriptorHeaderWords;
  for (std::size_t record_index = 0; record_index < record_count;
       ++record_index) {
    SemanticReductionRecord semantic;
    error = ReadSize(reader, &cursor, &semantic.rank);
    if (!error.success()) {
      return error;
    }
    if (semantic.rank > (reader.size() - cursor) / 4) {
      return Invalid("reduction semantic record is truncated");
    }
    uint64_t input_offset = 0;
    uint64_t output_offset = 0;
    for (uint64_t* value : {&input_offset, &output_offset,
                            &semantic.scale_real_bits,
                            &semantic.scale_imaginary_bits}) {
      error = ReadWord(reader, &cursor, value);
      if (!error.success()) {
        return error;
      }
    }
    if (input_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        output_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return Invalid("reduction offset exceeds signed host address domain");
    }
    semantic.input_offset = static_cast<int64_t>(input_offset);
    semantic.output_offset = static_cast<int64_t>(output_offset);
    uint64_t identity_scale = 0;
    error = ReadWord(reader, &cursor, &identity_scale);
    if (!error.success()) {
      return error;
    }
    if (identity_scale > 1) {
      return Invalid("invalid reduction identity mapping flag");
    }
    semantic.identity_scale = identity_scale != 0;
    error = ReadWord(reader, &cursor, &semantic.mapped_dtype);
    if (!error.success()) {
      return error;
    }
    if (semantic.mapped_dtype < kDtypeF32 || semantic.mapped_dtype > kDtypeU64) {
      return Invalid("unsupported reduction mapping dtype");
    }
    const uint64_t real_limit = ScalarWordLimit(semantic.mapped_dtype);
    if (semantic.scale_real_bits > real_limit ||
        semantic.scale_imaginary_bits > real_limit ||
        (!IsComplexDtype(semantic.mapped_dtype) &&
         semantic.scale_imaginary_bits != 0)) {
      return Invalid("reduction scale bits do not match mapped dtype");
    }
    semantic.shape.resize(semantic.rank);
    semantic.input_strides.resize(semantic.rank);
    semantic.output_strides.resize(semantic.rank);
    semantic.reduction_axes.resize(semantic.rank);
    semantic.logical_elements = 1;
    for (std::size_t axis = 0; axis < semantic.rank; ++axis) {
      error = ReadWord(reader, &cursor, &semantic.shape[axis]);
      if (!error.success() ||
          !CheckedMultiply(semantic.logical_elements, semantic.shape[axis],
                           &semantic.logical_elements)) {
        return Invalid("reduction logical element count overflow");
      }
    }
    for (auto* strides : {&semantic.input_strides,
                          &semantic.output_strides}) {
      for (std::size_t axis = 0; axis < semantic.rank; ++axis) {
        uint64_t word = 0;
        error = ReadWord(reader, &cursor, &word);
        if (!error.success()) {
          return error;
        }
        (*strides)[axis] = DecodeStrideWord(word);
        if ((*strides)[axis] == std::numeric_limits<int64_t>::min()) {
          return Invalid("reduction stride magnitude exceeds int64");
        }
      }
    }
    for (std::size_t axis = 0; axis < semantic.rank; ++axis) {
      uint64_t reduction_axis = 0;
      error = ReadWord(reader, &cursor, &reduction_axis);
      if (!error.success() || reduction_axis > 1) {
        return Invalid("reduction axis flag is invalid");
      }
      semantic.reduction_axes[axis] = reduction_axis != 0;
    }
    OptimizeSemanticReductionRecordForExecution(&semantic);
    ReductionRecord record;
    error = MaterializeReductionRecord(semantic, plan->input_size,
                                       plan->output_size, &record);
    if (!error.success()) {
      return error;
    }
    plan->records.push_back(record);
  }
  if (cursor != reader.size()) {
    return Invalid("reduction descriptor has trailing words");
  }
  return ffi::Error::Success();
}

int CompareReductionOutputMap(const ReductionRecord& left,
                              const ReductionRecord& right) {
  if (left.output_offset != right.output_offset) {
    return left.output_offset < right.output_offset ? -1 : 1;
  }
  if (left.output_count != right.output_count) {
    return left.output_count < right.output_count ? -1 : 1;
  }
  if (left.map_rank != right.map_rank) {
    return left.map_rank < right.map_rank ? -1 : 1;
  }
  for (std::size_t position = 0; position < left.map_rank; ++position) {
    const std::size_t left_axis = left.map_loop_order[position];
    const std::size_t right_axis = right.map_loop_order[position];
    if (left.shape[left_axis] != right.shape[right_axis]) {
      return left.shape[left_axis] < right.shape[right_axis] ? -1 : 1;
    }
    if (left.output_strides[left_axis] !=
        right.output_strides[right_axis]) {
      return left.output_strides[left_axis] <
                     right.output_strides[right_axis]
                 ? -1
                 : 1;
    }
  }
  return 0;
}

bool SameReductionOutputMap(const ReductionRecord& left,
                            const ReductionRecord& right) {
  return CompareReductionOutputMap(left, right) == 0;
}

ffi::Error BuildExactReductionGroups(ParsedReductionPlan* plan,
                                     bool require_grouped) {
  const auto unavailable = [&](const char* message) {
    plan->groups.clear();
    plan->grouped_chunks_per_batch = 0;
    plan->grouped_work_bytes_per_batch = 0;
    plan->grouped_output_owner = false;
    return require_grouped ? Invalid(message) : ffi::Error::Success();
  };
  plan->groups.clear();
  plan->grouped_chunks_per_batch = 0;
  plan->grouped_work_bytes_per_batch = 0;
  std::vector<std::size_t> record_order;
  record_order.reserve(plan->records.size());
  for (std::size_t record_index = 0;
       record_index < plan->records.size(); ++record_index) {
    const ReductionRecord& record = plan->records[record_index];
    if (record.output_count != 0 && record.reduction_count != 0) {
      record_order.push_back(record_index);
    }
  }
  std::sort(record_order.begin(), record_order.end(),
            [&](std::size_t left, std::size_t right) {
              const int comparison = CompareReductionOutputMap(
                  plan->records[left], plan->records[right]);
              return comparison < 0 ||
                     (comparison == 0 && left < right);
            });

  bool has_overlapping_group = false;
  const uint64_t source_item_size = DtypeItemSize(plan->input_dtype);
  for (const std::size_t record_index : record_order) {
    const ReductionRecord& record = plan->records[record_index];
    const bool starts_group =
        plan->groups.empty() ||
        !SameReductionOutputMap(
            plan->records[plan->groups.back().record_indices.front()],
            record);
    if (starts_group) {
      int64_t output_minimum = 0;
      int64_t output_maximum = 0;
      ffi::Error error = ReductionAddressBounds(
          record, false, record.map_axis_mask,
          &output_minimum, &output_maximum);
      if (!error.success()) {
        return error;
      }
      const uint64_t interval_size =
          static_cast<uint64_t>(output_maximum - output_minimum) + 1;
      if (interval_size != record.output_count) {
        return unavailable(
            "grouped reduction output map is not one physical interval");
      }
      plan->groups.emplace_back();
      plan->groups.back().output_minimum = output_minimum;
      plan->groups.back().output_maximum = output_maximum;
      plan->groups.back().output_count = record.output_count;
    } else {
      has_overlapping_group = true;
    }
    ReductionGroup& group = plan->groups.back();
    group.record_indices.push_back(record_index);
    if (!CheckedAdd(group.total_reduction_count,
                    record.reduction_count,
                    &group.total_reduction_count)) {
      return Invalid("grouped reduction fiber count overflow");
    }
  }
  if (record_order.size() < 2 || !has_overlapping_group) {
    return unavailable(
        "grouped reduction requires an exact overlapping output-map group");
  }

  std::vector<std::size_t> interval_order(plan->groups.size());
  std::iota(interval_order.begin(), interval_order.end(), 0);
  std::sort(interval_order.begin(), interval_order.end(),
            [&](std::size_t left, std::size_t right) {
              return std::make_pair(plan->groups[left].output_minimum, left) <
                     std::make_pair(plan->groups[right].output_minimum, right);
            });
  for (std::size_t position = 1; position < interval_order.size(); ++position) {
    const ReductionGroup& previous =
        plan->groups[interval_order[position - 1]];
    const ReductionGroup& current =
        plan->groups[interval_order[position]];
    if (previous.output_maximum >= current.output_minimum) {
      return unavailable(
          "grouped reduction output-map intervals are not disjoint");
    }
  }

  for (ReductionGroup& group : plan->groups) {
    uint64_t fiber_bytes = 0;
    if (!CheckedMultiply(group.total_reduction_count, source_item_size,
                         &fiber_bytes)) {
      return Invalid("grouped reduction fiber byte count overflow");
    }
    group.outputs_per_chunk = std::max<uint64_t>(
        1, fiber_bytes == 0 ? group.output_count
                            : plan->preferred_chunk_bytes / fiber_bytes);
    group.chunk_begin = plan->grouped_chunks_per_batch;
    group.chunk_count =
        1 + (group.output_count - 1) / group.outputs_per_chunk;
    if (!CheckedAdd(plan->grouped_chunks_per_batch, group.chunk_count,
                    &plan->grouped_chunks_per_batch)) {
      return Invalid("grouped reduction chunk count overflow");
    }
    uint64_t group_work_bytes = 0;
    if (!CheckedMultiply(group.output_count, fiber_bytes,
                         &group_work_bytes) ||
        !CheckedAdd(plan->grouped_work_bytes_per_batch, group_work_bytes,
                    &plan->grouped_work_bytes_per_batch)) {
      return Invalid("grouped reduction work byte count overflow");
    }
  }
  plan->grouped_output_owner = true;
  return ffi::Error::Success();
}

bool ShouldUseGroupedOutputOwner(const ParsedReductionPlan& plan,
                                 uint64_t batch_count) {
  if (!plan.grouped_output_owner || batch_count == 0) {
    return false;
  }
  uint64_t total_work_bytes = 0;
  uint64_t total_chunks = 0;
  if (!CheckedMultiply(batch_count, plan.grouped_work_bytes_per_batch,
                       &total_work_bytes) ||
      total_work_bytes < plan.parallel_minimum_bytes ||
      !CheckedMultiply(batch_count, plan.grouped_chunks_per_batch,
                       &total_chunks) ||
      total_chunks < 2) {
    return false;
  }
  const uint64_t source_item_size = DtypeItemSize(plan.input_dtype);
  for (const ReductionGroup& group : plan.groups) {
    uint64_t fiber_bytes = 0;
    uint64_t group_work_bytes = 0;
    uint64_t minimum_work_bytes = 0;
    if (!CheckedMultiply(group.total_reduction_count, source_item_size,
                         &fiber_bytes) ||
        !CheckedMultiply(group.output_count, fiber_bytes,
                         &group_work_bytes) ||
        !CheckedMultiply(group.chunk_count,
                         kGroupedMinimumWorkBytesPerChunk,
                         &minimum_work_bytes) ||
        group_work_bytes < minimum_work_bytes) {
      return false;
    }
  }
  return true;
}

ffi::Error CompileGeneratedReductionWork(ParsedReductionPlan* plan) {
  plan->generated_work.clear();
  plan->generated_ordered_records = false;
  if (plan->grouped_output_owner) {
    for (std::size_t group_index = 0; group_index < plan->groups.size();
         ++group_index) {
      const ReductionGroup& group = plan->groups[group_index];
      for (uint64_t chunk = 0; chunk < group.chunk_count; ++chunk) {
        const uint64_t output_start = chunk * group.outputs_per_chunk;
        plan->generated_work.push_back(GeneratedReductionWork{
            group_index,
            output_start,
            std::min(group.outputs_per_chunk,
                     group.output_count - output_start),
        });
      }
    }
    return ffi::Error::Success();
  }
  if (plan->records.size() != 1) {
    plan->generated_ordered_records = true;
    return ffi::Error::Success();
  }
  const ReductionRecord& record = plan->records.front();
  if (record.output_count == 0 || record.reduction_count == 0) {
    return ffi::Error::Success();
  }
  uint64_t fiber_bytes = 0;
  if (!CheckedMultiply(record.reduction_count,
                       DtypeItemSize(plan->input_dtype), &fiber_bytes)) {
    return Invalid("generated reduction fiber byte count overflow");
  }
  const uint64_t outputs_per_work = std::max<uint64_t>(
      1, fiber_bytes == 0 ? record.output_count
                          : plan->preferred_chunk_bytes / fiber_bytes);
  for (uint64_t output_start = 0; output_start < record.output_count;
       output_start += outputs_per_work) {
    plan->generated_work.push_back(GeneratedReductionWork{
        0,
        output_start,
        std::min(outputs_per_work, record.output_count - output_start),
    });
  }
  return ffi::Error::Success();
}
// Structured-reduction descriptor parsing, validation, grouping, and planning.
