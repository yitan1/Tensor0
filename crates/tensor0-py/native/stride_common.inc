std::atomic<uint64_t> native_call_count{0};
std::atomic<bool> observe_leaf_kernels_for_tests{false};
std::atomic<uint64_t> leaf_kernel_mask_for_tests{0};

void ObserveLeafKernel(uint64_t mask) {
  if (observe_leaf_kernels_for_tests.load(std::memory_order_relaxed)) {
    leaf_kernel_mask_for_tests.fetch_or(mask, std::memory_order_relaxed);
  }
}
std::atomic<uint64_t> last_worker_count{0};
std::atomic<uint64_t> last_available_worker_count{0};
std::atomic<uint64_t> worker_limit{std::numeric_limits<uint64_t>::max()};
std::atomic<bool> force_generic_for_tests{false};
std::atomic<bool> force_generated_baseline_for_tests{false};
std::atomic<uint64_t> reduction_fiber_parallel_mode_for_tests{
    std::numeric_limits<uint64_t>::max()};
std::atomic<uint64_t> last_reduction_fiber_chunks{0};
std::atomic<bool> last_grouped_output_owner{false};
std::atomic<bool> disable_f16_f16_contiguous_simd_for_tests{false};
std::atomic<bool> disable_f16_f32_contiguous_simd_for_tests{false};
std::atomic<bool> disable_f32_c64_contiguous_simd_for_tests{false};
std::atomic<uint64_t> prepared_instantiate_count{0};
std::atomic<uint64_t> prepared_execute_count{0};
std::atomic<uint64_t> prepared_live_state_count{0};
std::atomic<uint64_t> prepared_destroyed_state_count{0};
std::atomic<uint64_t> prepared_live_bytes{0};
std::atomic<uint64_t> prepared_last_state_bytes{0};
std::atomic<uint64_t> prepared_last_descriptor_bytes{0};
std::atomic<uint64_t> alias_last_base_pointer{0};
std::atomic<uint64_t> alias_last_source_pointer{0};
std::atomic<uint64_t> alias_last_factor_pointer{0};
std::atomic<uint64_t> alias_last_result_pointer{0};

struct SemanticRecord {
  std::size_t rank = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
  float scale = 1.0F;
  bool identity_scale = true;
  uint64_t mapped_dtype = 0;
  uint64_t logical_elements = 0;
  std::vector<uint64_t> shape;
  std::vector<int64_t> source_strides;
  std::vector<int64_t> destination_strides;
};

template <typename RecordType>
struct ParsedPlanData {
  uint64_t source_size = 0;
  uint64_t required_source_size = 0;
  uint64_t output_size = 0;
  uint64_t copied_elements = 0;
  uint64_t coverage = 0;
  uint64_t source_dtype = 0;
  uint64_t result_dtype = 0;
  uint64_t scalar_kind = 0;
  // Scalar computation dtype derived from the semantic dtype policy.
  uint64_t dtype = 0;
  std::vector<RecordType> records;
};

using ParsedSemanticPlan = ParsedPlanData<SemanticRecord>;
using ParsedPlan = ParsedPlanData<Record>;

enum class LeafKernel : uint8_t {
  // These values are leaf opcodes selected by CompileAffinePlan, not
  // independent plan representations or correctness paths.
  kContiguousRow,
  kContiguousInnerRows,
  kBroadcastContiguousRow,
  kScalarRow,
  kRank2ForwardTile,
  kRank2ReverseTile,
  kRank4TwoPairTile,
};

inline constexpr uint64_t kGeneratedEnable = UINT64_C(1) << 0;
inline constexpr uint64_t kGeneratedBlocking = UINT64_C(1) << 1;
inline constexpr uint64_t kGeneratedStridedOrder = UINT64_C(1) << 2;
inline constexpr uint64_t kGeneratedThreading = UINT64_C(1) << 4;
inline constexpr uint64_t kGeneratedProductionMode =
    kGeneratedEnable | kGeneratedBlocking | kGeneratedStridedOrder |
    kGeneratedThreading;

class DescriptorReader {
 public:
  explicit DescriptorReader(ffi::Span<const uint8_t> bytes) : bytes_(bytes) {}

  bool HasWholeWords() const {
    return bytes_.size() % sizeof(uint64_t) == 0;
  }

  std::size_t size() const { return bytes_.size() / sizeof(uint64_t); }

  bool Read(std::size_t index, uint64_t* value) const {
    if (index >= size()) {
      return false;
    }
    const std::size_t offset = index * sizeof(uint64_t);
    uint64_t decoded = 0;
    for (std::size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
      decoded |= static_cast<uint64_t>(bytes_[offset + byte]) << (8 * byte);
    }
    *value = decoded;
    return true;
  }

 private:
  ffi::Span<const uint8_t> bytes_;
};

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t* result) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

int64_t DecodeStrideWord(uint64_t value) {
  int64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

uint64_t AbsoluteStride(int64_t stride) {
  if (stride == std::numeric_limits<int64_t>::min()) {
    return UINT64_C(1) << 63;
  }
  return stride < 0 ? static_cast<uint64_t>(-stride)
                    : static_cast<uint64_t>(stride);
}

bool StrideEquals(int64_t stride, uint64_t value) {
  return value <=
             static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         stride == static_cast<int64_t>(value);
}

uint64_t DtypeComponentBytes(uint64_t dtype) {
  if (dtype == kDtypePred || dtype == kDtypeS8 || dtype == kDtypeU8) {
    return 1;
  }
  if (dtype == kDtypeF16 || dtype == kDtypeBF16 || dtype == kDtypeS16 ||
      dtype == kDtypeU16) {
    return 2;
  }
  if (dtype == kDtypeF32 || dtype == kDtypeC64 || dtype == kDtypeS32 ||
      dtype == kDtypeU32) {
    return 4;
  }
  if (dtype == kDtypeF64 || dtype == kDtypeC128 || dtype == kDtypeS64 ||
      dtype == kDtypeU64) {
    return 8;
  }
  return 0;
}

bool IsComplexDtype(uint64_t dtype) {
  return dtype == kDtypeC64 || dtype == kDtypeC128;
}

bool IsReductionDtype(uint64_t dtype) {
  return DtypeComponentBytes(dtype) != 0;
}

uint64_t ScalarWordLimit(uint64_t dtype) {
  const uint64_t bytes = DtypeComponentBytes(dtype);
  if (bytes == 0) {
    return 0;
  }
  if (bytes == sizeof(uint64_t)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return (UINT64_C(1) << (bytes * 8)) - 1;
}

bool PositiveStrideValue(int64_t stride, uint64_t* value) {
  if (stride <= 0) {
    return false;
  }
  *value = static_cast<uint64_t>(stride);
  return true;
}

bool CheckedMultiplyStride(int64_t stride, uint64_t extent,
                           int64_t* result) {
  uint64_t magnitude = 0;
  if (!CheckedMultiply(AbsoluteStride(stride), extent, &magnitude) ||
      magnitude >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  const int64_t signed_magnitude = static_cast<int64_t>(magnitude);
  *result = stride < 0 ? -signed_magnitude : signed_magnitude;
  return true;
}

void ComputeLoopOrder(Record* record) {
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    record->loop_axes_fastest_first[axis] = axis;
  }
  std::stable_sort(
      record->loop_axes_fastest_first.begin(),
      record->loop_axes_fastest_first.begin() + record->rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(
                   AbsoluteStride(record->destination_strides[left]),
                   AbsoluteStride(record->source_strides[left])) <
               std::make_pair(
                   AbsoluteStride(record->destination_strides[right]),
                   AbsoluteStride(record->source_strides[right]));
      });
}

ffi::Error Invalid(const std::string& message) {
  return ffi::Error::InvalidArgument("tensor0-stride: " + message);
}

ffi::Error ReadWord(const DescriptorReader& reader, std::size_t* cursor,
                    uint64_t* value) {
  if (!reader.Read(*cursor, value)) {
    return Invalid("descriptor is truncated");
  }
  ++*cursor;
  return ffi::Error::Success();
}

ffi::Error ReadSize(const DescriptorReader& reader, std::size_t* cursor,
                    std::size_t* value) {
  uint64_t raw = 0;
  ffi::Error error = ReadWord(reader, cursor, &raw);
  if (!error.success()) {
    return error;
  }
  if (raw > std::numeric_limits<std::size_t>::max() ||
      raw > reader.size()) {
    return Invalid("descriptor count exceeds its encoded size");
  }
  *value = static_cast<std::size_t>(raw);
  return ffi::Error::Success();
}

template <typename RecordType>
const auto& Strides(const RecordType& record, bool source) {
  return source ? record.source_strides : record.destination_strides;
}

template <typename RecordType>
int64_t Offset(const RecordType& record, bool source) {
  return source ? record.source_offset : record.destination_offset;
}

ffi::Error ParseSemanticRecords(
    const DescriptorReader& reader, std::size_t count,
    std::size_t* cursor, std::vector<SemanticRecord>* records) {
  records->reserve(count);
  for (std::size_t record_index = 0; record_index < count; ++record_index) {
    SemanticRecord record;
    ffi::Error error = ReadSize(reader, cursor, &record.rank);
    if (!error.success()) {
      return error;
    }
    uint64_t source_offset = 0;
    error = ReadWord(reader, cursor, &source_offset);
    if (!error.success()) {
      return error;
    }
    uint64_t destination_offset = 0;
    error = ReadWord(reader, cursor, &destination_offset);
    if (!error.success()) {
      return error;
    }
    if (source_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        destination_offset >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return Invalid("record offset exceeds the signed host address domain");
    }
    record.source_offset = static_cast<int64_t>(source_offset);
    record.destination_offset = static_cast<int64_t>(destination_offset);
    uint64_t scale_real_word = 0;
    uint64_t scale_imaginary_word = 0;
    error = ReadWord(reader, cursor, &scale_real_word);
    if (!error.success()) {
      return error;
    }
    error = ReadWord(reader, cursor, &scale_imaginary_word);
    if (!error.success()) {
      return error;
    }
    uint64_t identity_scale = 0;
    error = ReadWord(reader, cursor, &identity_scale);
    if (!error.success()) {
      return error;
    }
    if (identity_scale > 1) {
      return Invalid("invalid identity mapping flag");
    }
    record.identity_scale = identity_scale != 0;
    error = ReadWord(reader, cursor, &record.mapped_dtype);
    if (!error.success()) {
      return error;
    }
    if (record.mapped_dtype < kDtypeF32 || record.mapped_dtype > kDtypeU64) {
      return Invalid("unsupported mapping dtype");
    }
    const uint64_t real_limit = ScalarWordLimit(record.mapped_dtype);
    if (scale_real_word > real_limit ||
        scale_imaginary_word > real_limit ||
        (!IsComplexDtype(record.mapped_dtype) && scale_imaginary_word != 0)) {
      return Invalid("scale bits do not match descriptor dtype");
    }
    record.scale_real_bits = scale_real_word;
    record.scale_imaginary_bits = scale_imaginary_word;
    if (record.mapped_dtype == kDtypeF32) {
      const uint32_t bits = static_cast<uint32_t>(scale_real_word);
      std::memcpy(&record.scale, &bits, sizeof(float));
    }
    record.shape.resize(record.rank);
    record.source_strides.resize(record.rank);
    record.destination_strides.resize(record.rank);
    auto read_shape = [&](std::vector<uint64_t>* values) -> ffi::Error {
      for (std::size_t axis = 0; axis < record.rank; ++axis) {
        ffi::Error item_error = ReadWord(reader, cursor, &(*values)[axis]);
        if (!item_error.success()) {
          return item_error;
        }
      }
      return ffi::Error::Success();
    };
    error = read_shape(&record.shape);
    if (!error.success()) {
      return error;
    }
    auto read_strides = [&](std::vector<int64_t>* values) -> ffi::Error {
      for (std::size_t axis = 0; axis < record.rank; ++axis) {
        uint64_t word = 0;
        ffi::Error item_error = ReadWord(reader, cursor, &word);
        if (!item_error.success()) {
          return item_error;
        }
        (*values)[axis] = DecodeStrideWord(word);
        if ((*values)[axis] == std::numeric_limits<int64_t>::min()) {
          return Invalid("stride magnitude exceeds int64");
        }
      }
      return ffi::Error::Success();
    };
    for (auto* values :
         {&record.source_strides, &record.destination_strides}) {
      error = read_strides(values);
      if (!error.success()) {
        return error;
      }
    }

    uint64_t logical_elements = 1;
    for (std::size_t axis = 0; axis < record.rank; ++axis) {
      if (record.shape[axis] > 1 &&
          record.destination_strides[axis] == 0) {
        return Invalid("zero destination stride aliases a logical axis");
      }
      if (!CheckedMultiply(logical_elements, record.shape[axis],
                           &logical_elements)) {
        return Invalid("logical element count overflow");
      }
    }
    record.logical_elements = logical_elements;
    records->push_back(std::move(record));
  }
  return ffi::Error::Success();
}

uint64_t DtypeItemSize(uint64_t dtype) {
  const uint64_t component_bytes = DtypeComponentBytes(dtype);
  return IsComplexDtype(dtype) ? 2 * component_bytes : component_bytes;
}

template <ffi::DataType Dtype>
constexpr uint64_t DescriptorDtypeCode() {
  if constexpr (Dtype == ffi::PRED) {
    return kDtypePred;
  } else if constexpr (Dtype == ffi::S8) {
    return kDtypeS8;
  } else if constexpr (Dtype == ffi::S16) {
    return kDtypeS16;
  } else if constexpr (Dtype == ffi::S32) {
    return kDtypeS32;
  } else if constexpr (Dtype == ffi::S64) {
    return kDtypeS64;
  } else if constexpr (Dtype == ffi::U8) {
    return kDtypeU8;
  } else if constexpr (Dtype == ffi::U16) {
    return kDtypeU16;
  } else if constexpr (Dtype == ffi::U32) {
    return kDtypeU32;
  } else if constexpr (Dtype == ffi::U64) {
    return kDtypeU64;
  } else if constexpr (Dtype == ffi::F16) {
    return kDtypeF16;
  } else if constexpr (Dtype == ffi::BF16) {
    return kDtypeBF16;
  } else if constexpr (Dtype == ffi::F32) {
    return kDtypeF32;
  } else if constexpr (Dtype == ffi::F64) {
    return kDtypeF64;
  } else if constexpr (Dtype == ffi::C64) {
    return kDtypeC64;
  } else if constexpr (Dtype == ffi::C128) {
    return kDtypeC128;
  }
  return 0;
}

ffi::Error ValidateHostElements(uint64_t elements, uint64_t item_size,
                                const char* side) {
  if (item_size == 0) {
    return Invalid("descriptor dtype has no host element size");
  }
  const uint64_t size_limit =
      static_cast<uint64_t>(std::numeric_limits<std::size_t>::max()) /
      item_size;
  const uint64_t difference_limit =
      static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
      item_size;
  if (elements > size_limit || elements > difference_limit) {
    return Invalid(std::string(side) +
                   " size exceeds the host address domain");
  }
  return ffi::Error::Success();
}

template <typename RecordType>
ffi::Error AddressBounds(const RecordType& record, bool source,
                         int64_t* minimum, int64_t* maximum) {
  if (record.logical_elements == 0) {
    *minimum = Offset(record, source);
    *maximum = Offset(record, source);
    return ffi::Error::Success();
  }
  int64_t low = Offset(record, source);
  int64_t high = low;
  const auto& strides = Strides(record, source);
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    uint64_t magnitude = 0;
    if (!CheckedMultiply(record.shape[axis] - 1,
                         AbsoluteStride(strides[axis]), &magnitude) ||
        magnitude >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return Invalid(std::string(source ? "source" : "destination") +
                     " address arithmetic overflow");
    }
    if (strides[axis] < 0) {
      if (magnitude > static_cast<uint64_t>(low)) {
        return Invalid(std::string(source ? "source" : "destination") +
                       " address precedes storage");
      }
      low -= static_cast<int64_t>(magnitude);
    } else {
      if (magnitude > static_cast<uint64_t>(
                          std::numeric_limits<int64_t>::max() - high)) {
        return Invalid(std::string(source ? "source" : "destination") +
                       " address arithmetic overflow");
      }
      high += static_cast<int64_t>(magnitude);
    }
  }
  *minimum = low;
  *maximum = high;
  return ffi::Error::Success();
}

template <typename RecordType>
ffi::Error ValidateInjectiveView(const RecordType& record, bool source) {
  if (record.logical_elements == 0) {
    return ffi::Error::Success();
  }

  const auto& strides = Strides(record, source);
  std::vector<std::size_t> axes(record.rank);
  std::iota(axes.begin(), axes.end(), 0);
  std::stable_sort(
      axes.begin(), axes.end(),
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(strides[left]), left) <
               std::make_pair(AbsoluteStride(strides[right]), right);
      });

  uint64_t covered_span = 0;
  for (std::size_t position = 0; position < record.rank; ++position) {
    const std::size_t axis = axes[position];
    const uint64_t extent = record.shape[axis];
    const uint64_t stride = AbsoluteStride(strides[axis]);
    const bool broadcast = source && extent > 1 && strides[axis] == 0;
    if (extent <= 1 || broadcast) {
      continue;
    }
    if (stride <= covered_span) {
      return Invalid(std::string(source ? "source" : "destination") +
                     " strides do not prove an injective view");
    }
    uint64_t axis_span = 0;
    if (!CheckedMultiply(extent - 1, stride, &axis_span) ||
        !CheckedAdd(covered_span, axis_span, &covered_span)) {
      return Invalid("injectivity arithmetic overflow");
    }
  }
  return ffi::Error::Success();
}

template <typename RecordType>
ffi::Error ValidatePlan(ParsedPlanData<RecordType>* plan) {
  const uint64_t item_size = DtypeItemSize(plan->dtype);
  ffi::Error error =
      ValidateHostElements(plan->source_size, item_size, "source");
  if (!error.success()) {
    return error;
  }
  error = ValidateHostElements(plan->output_size, item_size, "destination");
  if (!error.success()) {
    return error;
  }
  if (plan->required_source_size > plan->source_size) {
    return Invalid("required source size exceeds bound source size");
  }

  uint64_t calculated_required_source_size = 0;
  uint64_t copied_elements = 0;
  for (const RecordType& record : plan->records) {
    int64_t source_minimum = 0;
    int64_t source_maximum = 0;
    int64_t destination_minimum = 0;
    int64_t destination_maximum = 0;
    error = ValidateInjectiveView(record, false);
    if (!error.success()) {
      return error;
    }
    error = AddressBounds(record, true, &source_minimum, &source_maximum);
    if (!error.success()) {
      return error;
    }
    error = AddressBounds(record, false, &destination_minimum,
                          &destination_maximum);
    if (!error.success()) {
      return error;
    }
    if (record.logical_elements == 0) {
      calculated_required_source_size =
          std::max(calculated_required_source_size,
                   static_cast<uint64_t>(record.source_offset));
      if (static_cast<uint64_t>(record.destination_offset) >
          plan->output_size) {
        return Invalid("empty destination offset exceeds output size");
      }
    } else {
      if (source_minimum < 0 || destination_minimum < 0) {
        return Invalid("record address precedes storage");
      }
      uint64_t source_end = 0;
      if (!CheckedAdd(static_cast<uint64_t>(source_maximum), 1,
                      &source_end)) {
        return Invalid("source end address overflow");
      }
      calculated_required_source_size =
          std::max(calculated_required_source_size, source_end);
      if (static_cast<uint64_t>(destination_maximum) >=
          plan->output_size) {
        return Invalid("destination address exceeds output size");
      }
    }
    if (!CheckedAdd(copied_elements, record.logical_elements,
                    &copied_elements)) {
      return Invalid("copied element count overflow");
    }
  }
  if (calculated_required_source_size != plan->required_source_size) {
    return Invalid("required source size mismatch");
  }
  if (plan->coverage == kCoverageCompleteUnique &&
      copied_elements != plan->output_size) {
    return Invalid("CompleteUnique copied element count mismatch");
  }
  if (plan->coverage == kCoveragePartialUniqueZeroFill &&
      copied_elements > plan->output_size) {
    return Invalid("PartialUnique copied element count exceeds output size");
  }
  plan->copied_elements = copied_elements;
  return ffi::Error::Success();
}

void OptimizeSemanticRecordForExecution(SemanticRecord* record) {
  if (record->logical_elements == 0 || record->rank == 0) {
    return;
  }

  std::vector<uint64_t> compacted_shape;
  std::vector<int64_t> compacted_source_strides;
  std::vector<int64_t> compacted_destination_strides;
  compacted_shape.reserve(record->rank);
  compacted_source_strides.reserve(record->rank);
  compacted_destination_strides.reserve(record->rank);
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    if (record->shape[axis] == 1) {
      continue;
    }
    compacted_shape.push_back(record->shape[axis]);
    compacted_source_strides.push_back(record->source_strides[axis]);
    compacted_destination_strides.push_back(
        record->destination_strides[axis]);
  }
  record->shape = std::move(compacted_shape);
  record->source_strides = std::move(compacted_source_strides);
  record->destination_strides = std::move(compacted_destination_strides);
  record->rank = record->shape.size();

  std::size_t axis = 0;
  while (axis + 1 < record->rank) {
    int64_t left_source_span = 0;
    int64_t left_destination_span = 0;
    int64_t right_source_span = 0;
    int64_t right_destination_span = 0;
    const bool left_fastest =
        CheckedMultiplyStride(record->source_strides[axis],
                              record->shape[axis], &left_source_span) &&
        CheckedMultiplyStride(record->destination_strides[axis],
                              record->shape[axis],
                              &left_destination_span) &&
        left_source_span == record->source_strides[axis + 1] &&
        left_destination_span ==
            record->destination_strides[axis + 1];
    const bool right_fastest =
        CheckedMultiplyStride(record->source_strides[axis + 1],
                              record->shape[axis + 1],
                              &right_source_span) &&
        CheckedMultiplyStride(record->destination_strides[axis + 1],
                              record->shape[axis + 1],
                              &right_destination_span) &&
        right_source_span == record->source_strides[axis] &&
        right_destination_span == record->destination_strides[axis];
    if (!left_fastest && !right_fastest) {
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
      record->source_strides[axis] =
          record->source_strides[axis + 1];
      record->destination_strides[axis] =
          record->destination_strides[axis + 1];
    }
    record->shape.erase(record->shape.begin() + axis + 1);
    record->source_strides.erase(record->source_strides.begin() + axis + 1);
    record->destination_strides.erase(
        record->destination_strides.begin() + axis + 1);
    --record->rank;
    if (axis != 0) {
      --axis;
    }
  }
}

ffi::Error MaterializePreparedRecord(const SemanticRecord& semantic,
                                     Record* record) {
  if (semantic.rank > kMaximumRank) {
    return Invalid("effective logical rank exceeds native limit " +
                   std::to_string(kMaximumRank));
  }
  record->rank = semantic.rank;
  record->source_offset = semantic.source_offset;
  record->destination_offset = semantic.destination_offset;
  record->scale_real_bits = semantic.scale_real_bits;
  record->scale_imaginary_bits = semantic.scale_imaginary_bits;
  record->scale = semantic.scale;
  record->identity_scale = semantic.identity_scale;
  record->mapped_dtype = semantic.mapped_dtype;
  record->logical_elements = semantic.logical_elements;
  for (std::size_t axis = 0; axis < semantic.rank; ++axis) {
    record->shape[axis] = semantic.shape[axis];
    record->source_strides[axis] = semantic.source_strides[axis];
    record->destination_strides[axis] = semantic.destination_strides[axis];
  }
  ComputeLoopOrder(record);
  return ffi::Error::Success();
}

ffi::Error ParseAndValidateDescriptor(const DescriptorReader& reader,
                                      ParsedPlan* plan) {
  if (!reader.HasWholeWords()) {
    return Invalid("descriptor byte length is not divisible by eight");
  }
  if (reader.size() < kHeaderWords) {
    return Invalid("descriptor is truncated");
  }
  std::array<uint64_t, kHeaderWords> header{};
  for (std::size_t index = 0; index < header.size(); ++index) {
    if (!reader.Read(index, &header[index])) {
      return Invalid("descriptor header is truncated");
    }
  }
  if (header[0] != kDescriptorMagic) {
    return Invalid("descriptor magic mismatch");
  }
  if (header[1] != kDescriptorVersion) {
    return Invalid("descriptor version mismatch");
  }
  if (header[2] != reader.size()) {
    return Invalid("descriptor word count mismatch");
  }
  if (header[7] != kCoverageCompleteUnique &&
      header[7] != kCoveragePartialUniqueZeroFill) {
    return Invalid("semantic witness coverage mode is invalid");
  }
  if (DtypeItemSize(header[8]) == 0 || DtypeItemSize(header[9]) == 0) {
    return Invalid("semantic descriptor operand dtype is invalid");
  }
  if (header[10] != kScalarForwardScaleCast &&
      header[10] != kScalarJaxTranspose) {
    return Invalid("semantic descriptor scalar policy is invalid");
  }
  if (header[11] != 0) {
    return Invalid("semantic descriptor execution flags are invalid");
  }
  ParsedSemanticPlan semantic;
  semantic.source_size = header[3];
  semantic.required_source_size = header[4];
  semantic.output_size = header[5];
  semantic.coverage = header[7];
  semantic.source_dtype = header[8];
  semantic.result_dtype = header[9];
  semantic.scalar_kind = header[10];
  semantic.dtype = semantic.scalar_kind == kScalarJaxTranspose
                       ? semantic.source_dtype
                       : semantic.result_dtype;
  if (header[6] > reader.size()) {
    return Invalid("record count exceeds descriptor size");
  }
  const std::size_t record_count = static_cast<std::size_t>(header[6]);

  std::size_t cursor = kHeaderWords;
  ffi::Error error = ParseSemanticRecords(
      reader, record_count, &cursor, &semantic.records);
  if (!error.success()) {
    return error;
  }
  if (cursor != reader.size()) {
    return Invalid("descriptor has trailing words");
  }
  error = ValidatePlan(&semantic);
  if (!error.success()) {
    return error;
  }

  plan->source_size = semantic.source_size;
  plan->required_source_size = semantic.required_source_size;
  plan->output_size = semantic.output_size;
  plan->copied_elements = semantic.copied_elements;
  plan->coverage = semantic.coverage;
  plan->source_dtype = semantic.source_dtype;
  plan->result_dtype = semantic.result_dtype;
  plan->scalar_kind = semantic.scalar_kind;
  plan->dtype = semantic.dtype;
  plan->records.clear();
  plan->records.reserve(semantic.records.size());
  for (SemanticRecord& semantic_record : semantic.records) {
    OptimizeSemanticRecordForExecution(&semantic_record);
    Record record;
    error = MaterializePreparedRecord(semantic_record, &record);
    if (!error.success()) {
      return error;
    }
    record.semantic_index = plan->records.size();
    plan->records.push_back(record);
  }
  return ffi::Error::Success();
}

template <typename Dimensions>
ffi::Error CountBufferElements(Dimensions dimensions, uint64_t* elements) {
  uint64_t count = 1;
  for (int64_t dimension : dimensions) {
    if (dimension < 0) {
      return Invalid("physical buffer size must be nonnegative");
    }
    if (!CheckedMultiply(count, static_cast<uint64_t>(dimension), &count)) {
      return Invalid("physical buffer element count overflow");
    }
  }
  *elements = count;
  return ffi::Error::Success();
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype>
ffi::Error ValidateBufferDimensions(ffi::Buffer<SourceDtype> source,
                                    ffi::ResultBuffer<ResultDtype> result,
                                    uint64_t source_size,
                                    uint64_t output_size,
                                    uint64_t* batch_count,
                                    bool allow_batched_rank = false) {
  const auto source_dims = source.dimensions();
  const auto result_dims = result->dimensions();
  if (!allow_batched_rank &&
      (source_dims.size() != 1 || result_dims.size() != 1)) {
    return Invalid("source and result must be rank-one physical buffers");
  }
  uint64_t source_elements = 0;
  uint64_t result_elements = 0;
  ffi::Error error = CountBufferElements(source_dims, &source_elements);
  if (!error.success()) {
    return error;
  }
  error = CountBufferElements(result_dims, &result_elements);
  if (!error.success()) {
    return error;
  }

  uint64_t batches = 0;
  bool batch_count_known = false;
  if (source_size == 0) {
    if (source_elements != 0) {
      return Invalid("actual flat storage size does not match descriptor");
    }
  } else {
    if (source_elements % source_size != 0) {
      return Invalid("actual flat storage size does not match descriptor");
    }
    batches = source_elements / source_size;
    batch_count_known = true;
  }

  if (output_size == 0) {
    if (result_elements != 0) {
      return Invalid("actual flat storage size does not match descriptor");
    }
  } else {
    if (result_elements % output_size != 0) {
      return Invalid("actual flat storage size does not match descriptor");
    }
    const uint64_t result_batches = result_elements / output_size;
    if (batch_count_known && result_batches != batches) {
      return Invalid("actual flat storage batch count does not match");
    }
    batches = result_batches;
    batch_count_known = true;
  }
  *batch_count = batch_count_known ? batches : 0;
  return ffi::Error::Success();
}

ffi::Error ValidateDisjointBuffers(const void* source, void* result,
                                   uint64_t source_elements,
                                   uint64_t result_elements,
                                   uint64_t item_size) {
  uint64_t source_bytes = 0;
  uint64_t result_bytes = 0;
  if (!CheckedMultiply(source_elements, item_size, &source_bytes) ||
      !CheckedMultiply(result_elements, item_size, &result_bytes)) {
    return Invalid("buffer byte count overflow");
  }
  if (source_bytes == 0 || result_bytes == 0) {
    return ffi::Error::Success();
  }
  const uintptr_t source_begin = reinterpret_cast<uintptr_t>(source);
  const uintptr_t result_begin = reinterpret_cast<uintptr_t>(result);
  if (source_bytes > std::numeric_limits<uintptr_t>::max() - source_begin ||
      result_bytes > std::numeric_limits<uintptr_t>::max() - result_begin) {
    return Invalid("buffer pointer range overflow");
  }
  const uintptr_t source_end = source_begin + source_bytes;
  const uintptr_t result_end = result_begin + result_bytes;
  if (source_begin < result_end && result_begin < source_end) {
    return Invalid("source and result buffers overlap");
  }
  return ffi::Error::Success();
}

bool IsCompactSameMapping(const Record& record) {
  if (record.rank == 0) {
    return true;
  }
  std::array<std::size_t, kMaximumRank> order{};
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    if (record.source_strides[axis] != record.destination_strides[axis] ||
        record.source_strides[axis] <= 0) {
      return false;
    }
    order[axis] = axis;
  }
  std::stable_sort(
      order.begin(), order.begin() + record.rank,
      [&](std::size_t left, std::size_t right) {
        return record.source_strides[left] < record.source_strides[right];
      });
  uint64_t expected = 1;
  for (std::size_t position = 0; position < record.rank; ++position) {
    const std::size_t axis = order[position];
    if (record.shape[axis] <= 1) {
      continue;
    }
    if (record.source_strides[axis] != static_cast<int64_t>(expected) ||
        !CheckedMultiply(expected, record.shape[axis], &expected)) {
      return false;
    }
  }
  return true;
}

bool IsRank2ForwardCandidate(const Record& record) {
  return record.rank == 2 && record.source_strides[0] == 1 &&
         StrideEquals(record.source_strides[1], record.shape[0]) &&
         record.destination_strides[1] == 1 &&
         StrideEquals(record.destination_strides[0], record.shape[1]);
}

bool IsRank2ReverseCandidate(const Record& record) {
  return record.rank == 2 && record.source_strides[1] == 1 &&
         StrideEquals(record.source_strides[0], record.shape[1]) &&
         record.destination_strides[0] == 1 &&
         StrideEquals(record.destination_strides[1], record.shape[0]);
}

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_TARGET
void Transpose8x8(__m256* values) {
  const __m256 t0 = _mm256_unpacklo_ps(values[0], values[1]);
  const __m256 t1 = _mm256_unpackhi_ps(values[0], values[1]);
  const __m256 t2 = _mm256_unpacklo_ps(values[2], values[3]);
  const __m256 t3 = _mm256_unpackhi_ps(values[2], values[3]);
  const __m256 t4 = _mm256_unpacklo_ps(values[4], values[5]);
  const __m256 t5 = _mm256_unpackhi_ps(values[4], values[5]);
  const __m256 t6 = _mm256_unpacklo_ps(values[6], values[7]);
  const __m256 t7 = _mm256_unpackhi_ps(values[6], values[7]);
  const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
  const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
  const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
  const __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
  const __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
  values[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
  values[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
  values[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
  values[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
  values[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
  values[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
  values[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
  values[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}


bool CpuSupportsAvx2() {
  static const bool supported = __builtin_cpu_supports("avx2");
  return supported;
}

bool CpuSupportsAvx2Fma() {
  static const bool supported =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return supported;
}

bool CpuSupportsAvx2F16c() {
  static const bool supported =
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
  return supported;
}
#endif

bool IsRank4TwoPairCandidate(const Record& record);

// Shared descriptor parsing, validation, and low-level affine helpers.
