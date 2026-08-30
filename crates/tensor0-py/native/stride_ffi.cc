#include "stride_descriptor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <complex>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TENSOR0_STRIDE_HAS_AVX2_TARGET 1
#define TENSOR0_STRIDE_AVX2_TARGET __attribute__((target("avx2")))
#define TENSOR0_STRIDE_AVX2_FMA_TARGET __attribute__((target("avx2,fma")))
#endif

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifndef TENSOR0_STRIDE_JAX_VERSION
#define TENSOR0_STRIDE_JAX_VERSION "unknown"
#endif

#ifndef TENSOR0_STRIDE_JAXLIB_VERSION
#define TENSOR0_STRIDE_JAXLIB_VERSION "unknown"
#endif

namespace ffi = xla::ffi;

namespace tensor0::stride {
namespace {

std::atomic<uint64_t> native_call_count{0};
std::atomic<uint64_t> last_worker_count{0};
std::atomic<uint64_t> last_available_worker_count{0};
std::atomic<uint64_t> worker_limit{std::numeric_limits<uint64_t>::max()};
std::atomic<uint64_t> prepared_instantiate_count{0};
std::atomic<uint64_t> prepared_execute_count{0};
std::atomic<uint64_t> prepared_live_state_count{0};
std::atomic<uint64_t> prepared_destroyed_state_count{0};
std::atomic<uint64_t> prepared_live_bytes{0};
std::atomic<uint64_t> prepared_last_state_bytes{0};
std::atomic<uint64_t> prepared_last_descriptor_bytes{0};

struct ParsedPlan {
  uint64_t source_size = 0;
  uint64_t required_source_size = 0;
  uint64_t output_size = 0;
  uint64_t copied_elements = 0;
  uint64_t coverage = 0;
  uint64_t source_dtype = 0;
  uint64_t result_dtype = 0;
  uint64_t scalar_kind = 0;
  // The embedded raw witness is encoded in the scalar operation dtype.
  uint64_t dtype = 0;
  std::vector<Record> records;
};

using AxisProvenance = std::vector<std::vector<std::size_t>>;

enum class RecordWorkKind : uint8_t {
  kCompact,
  kGeneric,
  kRank2Forward,
  kRank2Reverse,
  kRank4Axis0,
  kRank4TwoPair,
  kRank2SignedPermutation,
};

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

const std::array<int64_t, kMaximumRank>& Strides(const Record& record,
                                                 bool source) {
  return source ? record.source_strides : record.destination_strides;
}

int64_t Offset(const Record& record, bool source) {
  return source ? record.source_offset : record.destination_offset;
}

ffi::Error ParseRecords(const DescriptorReader& reader, uint64_t dtype,
                        std::size_t count,
                        std::size_t* cursor, std::vector<Record>* records) {
  records->reserve(count);
  for (std::size_t record_index = 0; record_index < count; ++record_index) {
    Record record;
    ffi::Error error = ReadSize(reader, cursor, &record.rank);
    if (!error.success()) {
      return error;
    }
    if (record.rank > kMaximumRank) {
      return Invalid("logical rank exceeds the native limit");
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
    const uint64_t real_limit = ScalarWordLimit(dtype);
    if (scale_real_word > real_limit ||
        scale_imaginary_word > real_limit ||
        (!IsComplexDtype(dtype) && scale_imaginary_word != 0)) {
      return Invalid("scale bits do not match descriptor dtype");
    }
    record.scale_real_bits = scale_real_word;
    record.scale_imaginary_bits = scale_imaginary_word;
    if (dtype == kDtypeF32) {
      const uint32_t bits = static_cast<uint32_t>(scale_real_word);
      std::memcpy(&record.scale, &bits, sizeof(float));
    }
    error = ReadWord(reader, cursor, &record.source_broadcast_axis_mask);
    if (!error.success()) {
      return error;
    }
    const uint64_t valid_broadcast_mask =
        record.rank == 0 ? 0 : (UINT64_C(1) << record.rank) - 1;
    if ((record.source_broadcast_axis_mask & ~valid_broadcast_mask) != 0) {
      return Invalid("source broadcast mask exceeds logical rank");
    }

    auto read_shape = [&](std::array<uint64_t, kMaximumRank>* values)
        -> ffi::Error {
      for (std::size_t axis = 0; axis < record.rank; ++axis) {
        ffi::Error item_error =
            ReadWord(reader, cursor, &(*values)[axis]);
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
    auto read_strides = [&](std::array<int64_t, kMaximumRank>* values)
        -> ffi::Error {
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
      const bool source_broadcast =
          (record.source_broadcast_axis_mask & (UINT64_C(1) << axis)) != 0;
      if (source_broadcast !=
          (record.shape[axis] > 1 && record.source_strides[axis] == 0)) {
        return Invalid(
            "source broadcast mask does not match zero source strides");
      }
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
    ComputeLoopOrder(&record);
    records->push_back(record);
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

ffi::Error AddressBounds(const Record& record, bool source,
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

ffi::Error ValidateInjectiveView(const Record& record, bool source) {
  if (record.logical_elements == 0) {
    return ffi::Error::Success();
  }

  const auto& strides = Strides(record, source);
  std::array<std::size_t, kMaximumRank> axes{};
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    axes[axis] = axis;
  }
  std::stable_sort(
      axes.begin(), axes.begin() + record.rank,
      [&](std::size_t left, std::size_t right) {
        return std::make_pair(AbsoluteStride(strides[left]), left) <
               std::make_pair(AbsoluteStride(strides[right]), right);
      });

  uint64_t covered_span = 0;
  for (std::size_t position = 0; position < record.rank; ++position) {
    const std::size_t axis = axes[position];
    const uint64_t extent = record.shape[axis];
    const uint64_t stride = AbsoluteStride(strides[axis]);
    const bool broadcast =
        source &&
        (record.source_broadcast_axis_mask & (UINT64_C(1) << axis)) != 0;
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

ffi::Error ValidatePlan(ParsedPlan* plan) {
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
  for (const Record& record : plan->records) {
    int64_t source_minimum = 0;
    int64_t source_maximum = 0;
    int64_t destination_minimum = 0;
    int64_t destination_maximum = 0;
    error = ValidateInjectiveView(record, true);
    if (!error.success()) {
      return error;
    }
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

void OptimizeRecordForExecution(Record* record,
                                AxisProvenance* provenance = nullptr) {
  if (provenance != nullptr) {
    provenance->clear();
    provenance->reserve(record->rank);
    for (std::size_t axis = 0; axis < record->rank; ++axis) {
      provenance->push_back({axis});
    }
  }
  if (record->logical_elements == 0 || record->rank == 0) {
    return;
  }

  std::size_t compacted_rank = 0;
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    if (record->shape[axis] == 1) {
      continue;
    }
    record->shape[compacted_rank] = record->shape[axis];
    record->source_strides[compacted_rank] =
        record->source_strides[axis];
    record->destination_strides[compacted_rank] =
        record->destination_strides[axis];
    if (provenance != nullptr && compacted_rank != axis) {
      (*provenance)[compacted_rank] = std::move((*provenance)[axis]);
    }
    ++compacted_rank;
  }
  record->rank = compacted_rank;
  if (provenance != nullptr) {
    provenance->resize(compacted_rank);
  }

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
    if (provenance != nullptr) {
      std::vector<std::size_t> merged;
      merged.reserve((*provenance)[axis].size() +
                     (*provenance)[axis + 1].size());
      const auto append = [&](const std::vector<std::size_t>& axes) {
        merged.insert(merged.end(), axes.begin(), axes.end());
      };
      if (right_fastest) {
        append((*provenance)[axis + 1]);
        append((*provenance)[axis]);
      } else {
        append((*provenance)[axis]);
        append((*provenance)[axis + 1]);
      }
      (*provenance)[axis] = std::move(merged);
      provenance->erase(provenance->begin() + axis + 1);
    }
    for (std::size_t tail = axis + 1; tail + 1 < record->rank; ++tail) {
      record->shape[tail] = record->shape[tail + 1];
      record->source_strides[tail] =
          record->source_strides[tail + 1];
      record->destination_strides[tail] =
          record->destination_strides[tail + 1];
    }
    --record->rank;
    if (axis != 0) {
      --axis;
    }
  }

  record->source_broadcast_axis_mask = 0;
  for (std::size_t broadcast_axis = 0; broadcast_axis < record->rank;
       ++broadcast_axis) {
    if (record->shape[broadcast_axis] > 1 &&
        record->source_strides[broadcast_axis] == 0) {
      record->source_broadcast_axis_mask |=
          UINT64_C(1) << broadcast_axis;
    }
  }
  ComputeLoopOrder(record);
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
  if (DtypeItemSize(header[8]) == 0) {
    return Invalid("semantic witness dtype is invalid");
  }
  plan->source_size = header[3];
  plan->required_source_size = header[4];
  plan->output_size = header[5];
  plan->coverage = header[7];
  plan->dtype = header[8];
  if (header[6] > reader.size()) {
    return Invalid("record count exceeds descriptor size");
  }
  const std::size_t record_count = static_cast<std::size_t>(header[6]);

  std::size_t cursor = kHeaderWords;
  ffi::Error error = ParseRecords(reader, plan->dtype, record_count, &cursor,
                                  &plan->records);
  if (!error.success()) {
    return error;
  }
  if (cursor != reader.size()) {
    return Invalid("descriptor has trailing words");
  }
  error = ValidatePlan(plan);
  if (!error.success()) {
    return error;
  }
  return ffi::Error::Success();
}

ffi::Error ParseDescriptor(const DescriptorReader& reader, ParsedPlan* plan) {
  ffi::Error error = ParseAndValidateDescriptor(reader, plan);
  if (!error.success()) {
    return error;
  }
  for (Record& record : plan->records) {
    OptimizeRecordForExecution(&record);
  }
  return ffi::Error::Success();
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype>
ffi::Error ValidateBufferDimensions(ffi::Buffer<SourceDtype> source,
                                    ffi::ResultBuffer<ResultDtype> result,
                                    uint64_t source_size,
                                    uint64_t output_size,
                                    uint64_t* batch_count) {
  const auto source_dims = source.dimensions();
  const auto result_dims = result->dimensions();
  if (source_dims.size() != 1 || result_dims.size() != 1) {
    return Invalid("source and result must be rank-one physical buffers");
  }
  if (source_dims[0] < 0 || result_dims[0] < 0) {
    return Invalid("physical buffer size must be nonnegative");
  }
  const uint64_t source_elements =
      static_cast<uint64_t>(source_dims[0]);
  const uint64_t result_elements =
      static_cast<uint64_t>(result_dims[0]);

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

void ExecuteCompact(const Record& record, const float* source, float* result) {
  const float* input = source + record.source_offset;
  float* output = result + record.destination_offset;
  if (record.scale == 1.0F) {
    std::memcpy(output, input,
                static_cast<std::size_t>(record.logical_elements) *
                    sizeof(float));
    return;
  }
  for (uint64_t element = 0; element < record.logical_elements; ++element) {
    output[element] = record.scale * input[element];
  }
}

void ExecuteRank2TransposeColumns(const Record& record, const float* source,
                                  float* result, uint64_t first_column,
                                  uint64_t column_count) {
  constexpr uint64_t tile = 32;
  const uint64_t rows = record.shape[0];
  const uint64_t column_stop_limit = first_column + column_count;
  for (uint64_t column_start = first_column;
       column_start < column_stop_limit;
       column_start += tile) {
    const uint64_t column_stop =
        std::min(column_stop_limit, column_start + tile);
    for (uint64_t row_start = 0; row_start < rows; row_start += tile) {
      const uint64_t row_stop = std::min(rows, row_start + tile);
      for (uint64_t column = column_start; column < column_stop; ++column) {
        uint64_t source_index =
            record.source_offset +
            row_start * record.source_strides[0] +
            column * record.source_strides[1];
        uint64_t destination_index =
            record.destination_offset +
            row_start * record.destination_strides[0] +
            column * record.destination_strides[1];
        for (uint64_t row = row_start; row < row_stop; ++row) {
          result[destination_index] = record.scale * source[source_index];
          source_index += record.source_strides[0];
          destination_index += record.destination_strides[0];
        }
      }
    }
  }
}

void ExecuteRank2Transpose(const Record& record, const float* source,
                           float* result) {
  ExecuteRank2TransposeColumns(record, source, result, 0, record.shape[1]);
}

void ExecuteRank2TransposeRows(const Record& record, const float* source,
                               float* result, uint64_t first_row,
                               uint64_t row_count) {
  constexpr uint64_t tile = 32;
  const uint64_t columns = record.shape[1];
  const uint64_t row_stop_limit = first_row + row_count;
  for (uint64_t row_start = first_row; row_start < row_stop_limit;
       row_start += tile) {
    const uint64_t row_stop =
        std::min(row_stop_limit, row_start + tile);
    for (uint64_t column_start = 0; column_start < columns;
         column_start += tile) {
      const uint64_t column_stop =
          std::min(columns, column_start + tile);
      for (uint64_t row = row_start; row < row_stop; ++row) {
        uint64_t source_index =
            record.source_offset +
            row * record.source_strides[0] +
            column_start * record.source_strides[1];
        uint64_t destination_index =
            record.destination_offset +
            row * record.destination_strides[0] +
            column_start * record.destination_strides[1];
        for (uint64_t column = column_start; column < column_stop; ++column) {
          result[destination_index] = record.scale * source[source_index];
          source_index += record.source_strides[1];
          destination_index += record.destination_strides[1];
        }
      }
    }
  }
}

void ExecuteRank2TransposeReverse(const Record& record,
                                  const float* source, float* result) {
  ExecuteRank2TransposeRows(record, source, result, 0, record.shape[0]);
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

template <bool kUnitScale>
TENSOR0_STRIDE_AVX2_TARGET
bool ExecuteRank4TiledAvx2(const Record& record, const float* source,
                           float* result) {
  if (record.shape[1] % 8 != 0 || record.shape[3] % 4 != 0 ||
      record.shape[3] < 8) {
    return false;
  }
  const __m256 scale8 = _mm256_set1_ps(record.scale);
  const __m128 scale4 = _mm_set1_ps(record.scale);
  const bool forward = record.source_strides[1] == 1 &&
                       record.destination_strides[3] == 1;
  const bool reverse = record.source_strides[3] == 1 &&
                       record.destination_strides[1] == 1;
  if (!forward && !reverse) {
    return false;
  }

  for (uint64_t i0 = 0; i0 < record.shape[0]; ++i0) {
    const uint64_t source0 =
        record.source_offset + i0 * record.source_strides[0];
    const uint64_t destination0 =
        record.destination_offset + i0 * record.destination_strides[0];
    for (uint64_t i2 = 0; i2 < record.shape[2]; ++i2) {
      const uint64_t source2 =
          source0 + i2 * record.source_strides[2];
      const uint64_t destination2 =
          destination0 + i2 * record.destination_strides[2];
      if (forward) {
        uint64_t i3 = 0;
        for (; i3 + 8 <= record.shape[3]; i3 += 8) {
          for (uint64_t i1 = 0; i1 < record.shape[1]; i1 += 8) {
            __m256 values[8];
            for (uint64_t row = 0; row < 8; ++row) {
              values[row] = _mm256_loadu_ps(
                  source + source2 +
                  (i3 + row) * record.source_strides[3] + i1);
              if constexpr (!kUnitScale) {
                values[row] = _mm256_mul_ps(values[row], scale8);
              }
            }
            Transpose8x8(values);
            for (uint64_t row = 0; row < 8; ++row) {
              _mm256_storeu_ps(
                  result + destination2 +
                      (i1 + row) * record.destination_strides[1] + i3,
                  values[row]);
            }
          }
        }
        if (i3 < record.shape[3]) {
          for (uint64_t i1 = 0; i1 < record.shape[1]; i1 += 4) {
            __m128 values[4];
            for (uint64_t row = 0; row < 4; ++row) {
              values[row] = _mm_loadu_ps(
                  source + source2 +
                  (i3 + row) * record.source_strides[3] + i1);
              if constexpr (!kUnitScale) {
                values[row] = _mm_mul_ps(values[row], scale4);
              }
            }
            _MM_TRANSPOSE4_PS(values[0], values[1], values[2], values[3]);
            for (uint64_t row = 0; row < 4; ++row) {
              _mm_storeu_ps(
                  result + destination2 +
                      (i1 + row) * record.destination_strides[1] + i3,
                  values[row]);
            }
          }
        }
      } else {
        for (uint64_t i1 = 0; i1 < record.shape[1]; i1 += 8) {
          uint64_t i3 = 0;
          for (; i3 + 8 <= record.shape[3]; i3 += 8) {
            __m256 values[8];
            for (uint64_t row = 0; row < 8; ++row) {
              values[row] = _mm256_loadu_ps(
                  source + source2 +
                  (i1 + row) * record.source_strides[1] + i3);
              if constexpr (!kUnitScale) {
                values[row] = _mm256_mul_ps(values[row], scale8);
              }
            }
            Transpose8x8(values);
            for (uint64_t row = 0; row < 8; ++row) {
              _mm256_storeu_ps(
                  result + destination2 +
                      (i3 + row) * record.destination_strides[3] + i1,
                  values[row]);
            }
          }
          if (i3 < record.shape[3]) {
            for (uint64_t i1_tail = i1; i1_tail < i1 + 8;
                 i1_tail += 4) {
              __m128 values[4];
              for (uint64_t row = 0; row < 4; ++row) {
                values[row] = _mm_loadu_ps(
                    source + source2 +
                    (i1_tail + row) * record.source_strides[1] + i3);
                if constexpr (!kUnitScale) {
                  values[row] = _mm_mul_ps(values[row], scale4);
                }
              }
              _MM_TRANSPOSE4_PS(values[0], values[1], values[2], values[3]);
              for (uint64_t row = 0; row < 4; ++row) {
                _mm_storeu_ps(
                    result + destination2 +
                        (i3 + row) * record.destination_strides[3] +
                        i1_tail,
                    values[row]);
              }
            }
          }
        }
      }
    }
  }
  return true;
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
#endif

bool IsRank4TwoPairCandidate(const Record& record);

template <bool kUnitScale>
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_TARGET
#endif
bool ExecuteRank4TwoPair(const Record& record, const float* source,
                         float* result, uint64_t first_axis0,
                         uint64_t axis0_count) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (!CpuSupportsAvx2() || !IsRank4TwoPairCandidate(record) ||
      first_axis0 > record.shape[0] ||
      axis0_count > record.shape[0] - first_axis0) {
    return false;
  }
  const __m256 scale = _mm256_set1_ps(record.scale);
  const uint64_t axis0_stop = first_axis0 + axis0_count;
  for (uint64_t i0 = first_axis0; i0 < axis0_stop; ++i0) {
    const uint64_t source0 =
        record.source_offset + i0 * record.source_strides[0];
    const uint64_t destination0 =
        record.destination_offset + i0 * record.destination_strides[0];
    for (uint64_t i1 = 0; i1 < record.shape[1]; ++i1) {
      const uint64_t source1 = source0 + i1 * record.source_strides[1];
      const uint64_t destination1 =
          destination0 + i1 * record.destination_strides[1];
      for (uint64_t i3 = 0; i3 < record.shape[3]; i3 += 8) {
        for (uint64_t i2 = 0; i2 < record.shape[2]; i2 += 8) {
          __m256 values[8];
          for (uint64_t row = 0; row < 8; ++row) {
            values[row] = _mm256_loadu_ps(
                source + source1 +
                (i3 + row) * record.source_strides[3] + i2);
            if constexpr (!kUnitScale) {
              values[row] = _mm256_mul_ps(values[row], scale);
            }
          }
          Transpose8x8(values);
          for (uint64_t row = 0; row < 8; ++row) {
            _mm256_storeu_ps(
                result + destination1 +
                    (i2 + row) * record.destination_strides[2] + i3,
                values[row]);
          }
        }
      }
    }
  }
  return true;
#else
  static_cast<void>(record);
  static_cast<void>(source);
  static_cast<void>(result);
  return false;
#endif
}

template <bool kUnitScale>
bool ExecuteRank4TwoPair(const Record& record, const float* source,
                         float* result) {
  return ExecuteRank4TwoPair<kUnitScale>(
      record, source, result, 0, record.shape[0]);
}

template <bool kUnitScale>
bool ExecuteRank4Tiled(const Record& record, const float* source,
                       float* result) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2() &&
      ExecuteRank4TiledAvx2<kUnitScale>(record, source, result)) {
    return true;
  }
#endif
#if defined(__SSE__)
  const bool transpose_4x4 =
      record.shape[1] % 4 == 0 && record.shape[3] % 4 == 0 &&
      record.source_strides[1] == 1 &&
      record.destination_strides[3] == 1;
  if (transpose_4x4) {
    const __m128 scale = _mm_set1_ps(record.scale);
    for (uint64_t i0 = 0; i0 < record.shape[0]; ++i0) {
      const uint64_t source0 =
          record.source_offset + i0 * record.source_strides[0];
      const uint64_t destination0 =
          record.destination_offset + i0 * record.destination_strides[0];
      for (uint64_t i2 = 0; i2 < record.shape[2]; ++i2) {
        const uint64_t source2 =
            source0 + i2 * record.source_strides[2];
        const uint64_t destination2 =
            destination0 + i2 * record.destination_strides[2];
        for (uint64_t i3 = 0; i3 < record.shape[3]; i3 += 4) {
          for (uint64_t i1 = 0; i1 < record.shape[1]; i1 += 4) {
            __m128 values0 = _mm_loadu_ps(
                source + source2 +
                (i3 + 0) * record.source_strides[3] + i1);
            __m128 values1 = _mm_loadu_ps(
                source + source2 +
                (i3 + 1) * record.source_strides[3] + i1);
            __m128 values2 = _mm_loadu_ps(
                source + source2 +
                (i3 + 2) * record.source_strides[3] + i1);
            __m128 values3 = _mm_loadu_ps(
                source + source2 +
                (i3 + 3) * record.source_strides[3] + i1);
            if constexpr (!kUnitScale) {
              values0 = _mm_mul_ps(values0, scale);
              values1 = _mm_mul_ps(values1, scale);
              values2 = _mm_mul_ps(values2, scale);
              values3 = _mm_mul_ps(values3, scale);
            }
            _MM_TRANSPOSE4_PS(values0, values1, values2, values3);
            _mm_storeu_ps(
                result + destination2 +
                    (i1 + 0) * record.destination_strides[1] + i3,
                values0);
            _mm_storeu_ps(
                result + destination2 +
                    (i1 + 1) * record.destination_strides[1] + i3,
                values1);
            _mm_storeu_ps(
                result + destination2 +
                    (i1 + 2) * record.destination_strides[1] + i3,
                values2);
            _mm_storeu_ps(
                result + destination2 +
                    (i1 + 3) * record.destination_strides[1] + i3,
                values3);
          }
        }
      }
    }
    return true;
  }

  const bool reverse_transpose_4x4 =
      record.shape[1] % 4 == 0 && record.shape[3] % 4 == 0 &&
      record.source_strides[3] == 1 &&
      record.destination_strides[1] == 1;
  if (reverse_transpose_4x4) {
    const __m128 scale = _mm_set1_ps(record.scale);
    for (uint64_t i0 = 0; i0 < record.shape[0]; ++i0) {
      const uint64_t source0 =
          record.source_offset + i0 * record.source_strides[0];
      const uint64_t destination0 =
          record.destination_offset + i0 * record.destination_strides[0];
      for (uint64_t i2 = 0; i2 < record.shape[2]; ++i2) {
        const uint64_t source2 =
            source0 + i2 * record.source_strides[2];
        const uint64_t destination2 =
            destination0 + i2 * record.destination_strides[2];
        for (uint64_t i1 = 0; i1 < record.shape[1]; i1 += 4) {
          for (uint64_t i3 = 0; i3 < record.shape[3]; i3 += 4) {
            __m128 values0 = _mm_loadu_ps(
                source + source2 +
                (i1 + 0) * record.source_strides[1] + i3);
            __m128 values1 = _mm_loadu_ps(
                source + source2 +
                (i1 + 1) * record.source_strides[1] + i3);
            __m128 values2 = _mm_loadu_ps(
                source + source2 +
                (i1 + 2) * record.source_strides[1] + i3);
            __m128 values3 = _mm_loadu_ps(
                source + source2 +
                (i1 + 3) * record.source_strides[1] + i3);
            if constexpr (!kUnitScale) {
              values0 = _mm_mul_ps(values0, scale);
              values1 = _mm_mul_ps(values1, scale);
              values2 = _mm_mul_ps(values2, scale);
              values3 = _mm_mul_ps(values3, scale);
            }
            _MM_TRANSPOSE4_PS(values0, values1, values2, values3);
            _mm_storeu_ps(
                result + destination2 +
                    (i3 + 0) * record.destination_strides[3] + i1,
                values0);
            _mm_storeu_ps(
                result + destination2 +
                    (i3 + 1) * record.destination_strides[3] + i1,
                values1);
            _mm_storeu_ps(
                result + destination2 +
                    (i3 + 2) * record.destination_strides[3] + i1,
                values2);
            _mm_storeu_ps(
                result + destination2 +
                    (i3 + 3) * record.destination_strides[3] + i1,
                values3);
          }
        }
      }
    }
    return true;
  }
#else
  static_cast<void>(record);
  static_cast<void>(source);
  static_cast<void>(result);
#endif
  return false;
}

uint64_t GenericRowCount(const Record& record) {
  return record.logical_elements /
         record.shape[record.loop_axes_fastest_first[0]];
}

template <bool kUnitScale, typename SourceElement, typename ResultElement,
          typename ScalarOp>
void ExecuteGenericRecordRangeImpl(const Record& record,
                                   const SourceElement* source,
                                   ResultElement* result,
                                   uint64_t row_start,
                                   uint64_t row_count,
                                   ScalarOp scalar_op) {
  if (row_count == 0) {
    return;
  }
  const std::size_t fastest_axis = record.loop_axes_fastest_first[0];
  const uint64_t inner_elements = record.shape[fastest_axis];
  const int64_t source_inner_stride =
      record.source_strides[fastest_axis];
  const int64_t destination_inner_stride =
      record.destination_strides[fastest_axis];
  std::array<uint64_t, kMaximumRank> coordinates{};
  int64_t source_base = record.source_offset;
  int64_t destination_base = record.destination_offset;
  uint64_t remaining = row_start;
  for (std::size_t position = 1; position < record.rank; ++position) {
    const std::size_t axis = record.loop_axes_fastest_first[position];
    coordinates[axis] = remaining % record.shape[axis];
    remaining /= record.shape[axis];
    source_base += static_cast<int64_t>(coordinates[axis]) *
                   record.source_strides[axis];
    destination_base +=
        static_cast<int64_t>(coordinates[axis]) *
        record.destination_strides[axis];
  }

  for (uint64_t row = 0; row < row_count; ++row) {
    int64_t source_index = source_base;
    int64_t destination_index = destination_base;
    for (uint64_t inner = 0; inner < inner_elements; ++inner) {
      if constexpr (kUnitScale) {
        result[destination_index] = source[source_index];
      } else {
        result[destination_index] =
            scalar_op(source[source_index], record);
      }
      source_index += source_inner_stride;
      destination_index += destination_inner_stride;
    }
    if (row + 1 == row_count) {
      break;
    }
    for (std::size_t position = 1; position < record.rank; ++position) {
      const std::size_t axis = record.loop_axes_fastest_first[position];
      if (coordinates[axis] + 1 < record.shape[axis]) {
        ++coordinates[axis];
        source_base += record.source_strides[axis];
        destination_base += record.destination_strides[axis];
        break;
      }
      source_base -= static_cast<int64_t>(coordinates[axis]) *
                     record.source_strides[axis];
      destination_base -=
          static_cast<int64_t>(coordinates[axis]) *
          record.destination_strides[axis];
      coordinates[axis] = 0;
    }
  }
}

void ExecuteGeneric(const Record& record, const float* source, float* result) {
  if (record.rank == 0) {
    result[record.destination_offset] =
        record.scale * source[record.source_offset];
    return;
  }

  if (record.logical_elements == 0) {
    return;
  }
  const auto scale = [](float value, const Record& item) {
    return item.scale * value;
  };
  const uint64_t row_count = GenericRowCount(record);
  if (record.scale == 1.0F) {
    ExecuteGenericRecordRangeImpl<true>(record, source, result, 0,
                                        row_count, scale);
  } else {
    ExecuteGenericRecordRangeImpl<false>(record, source, result, 0,
                                         row_count, scale);
  }
}

bool IsRank2SignedPermutationCandidate(const Record& record) {
  if (record.rank != 2 || record.shape[0] <= 1 || record.shape[1] <= 1 ||
      record.shape[0] >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      record.shape[1] >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  uint64_t source_offset = 0;
  return CheckedMultiply(record.shape[0] - 1, record.shape[1],
                         &source_offset) &&
         record.source_strides[0] ==
             -static_cast<int64_t>(record.shape[1]) &&
         record.source_strides[1] == 1 &&
         record.source_offset == static_cast<int64_t>(source_offset) &&
         record.destination_strides[0] == 1 &&
         record.destination_strides[1] ==
             static_cast<int64_t>(record.shape[0]) &&
         record.destination_offset == 0;
}

template <bool kUnitScale>
void ExecuteRank2SignedPermutationRange(const Record& record,
                                        const float* source, float* result,
                                        uint64_t unit_start,
                                        uint64_t unit_count) {
  const uint64_t rows = record.shape[0];
  const uint64_t columns = record.shape[1];
  if (rows >= columns) {
    constexpr uint64_t kTile = 16;
    const uint64_t column_limit = unit_start + unit_count;
    for (uint64_t column_start = unit_start;
         column_start < column_limit; column_start += kTile) {
      const uint64_t column_end =
          std::min(column_limit, column_start + kTile);
      for (uint64_t row_start = 0; row_start < rows; row_start += kTile) {
        const uint64_t row_end = std::min(rows, row_start + kTile);
        for (uint64_t column = column_start; column < column_end; ++column) {
          float* destination_column = result + column * rows;
          for (uint64_t row = row_start; row < row_end; ++row) {
            if constexpr (kUnitScale) {
              destination_column[row] =
                  source[(rows - 1 - row) * columns + column];
            } else {
              destination_column[row] =
                  record.scale *
                  source[(rows - 1 - row) * columns + column];
            }
          }
        }
      }
    }
    return;
  }

  constexpr uint64_t kTile = 8;
  const uint64_t row_limit = unit_start + unit_count;
  for (uint64_t row_start = unit_start; row_start < row_limit;
       row_start += kTile) {
    const uint64_t row_end = std::min(row_limit, row_start + kTile);
    for (uint64_t column_start = 0; column_start < columns;
         column_start += kTile) {
      const uint64_t column_end =
          std::min(columns, column_start + kTile);
      for (uint64_t row = row_start; row < row_end; ++row) {
        const float* source_row = source + (rows - 1 - row) * columns;
        for (uint64_t column = column_start; column < column_end; ++column) {
          if constexpr (kUnitScale) {
            result[column * rows + row] = source_row[column];
          } else {
            result[column * rows + row] = record.scale * source_row[column];
          }
        }
      }
    }
  }
}

void ExecuteRank2SignedPermutation(const Record& record,
                                   const float* source, float* result) {
  const uint64_t unit_count =
      record.shape[0] >= record.shape[1] ? record.shape[1] : record.shape[0];
  if (record.scale == 1.0F) {
    ExecuteRank2SignedPermutationRange<true>(record, source, result, 0,
                                             unit_count);
  } else {
    ExecuteRank2SignedPermutationRange<false>(record, source, result, 0,
                                              unit_count);
  }
}

void ExecuteRecord(const Record& record, const float* source, float* result) {
  if (record.logical_elements == 0) {
    return;
  }
  if (IsCompactSameMapping(record)) {
    ExecuteCompact(record, source, result);
    return;
  }
  if (IsRank2SignedPermutationCandidate(record)) {
    ExecuteRank2SignedPermutation(record, source, result);
    return;
  }
  if (IsRank2ForwardCandidate(record)) {
    ExecuteRank2Transpose(record, source, result);
    return;
  }
  if (IsRank2ReverseCandidate(record)) {
    ExecuteRank2TransposeReverse(record, source, result);
    return;
  }
  if (record.rank == 4) {
    const bool two_pair_executed =
        record.scale == 1.0F
            ? ExecuteRank4TwoPair<true>(record, source, result)
            : ExecuteRank4TwoPair<false>(record, source, result);
    if (two_pair_executed) {
      return;
    }
    const bool executed =
        record.scale == 1.0F
            ? ExecuteRank4Tiled<true>(record, source, result)
            : ExecuteRank4Tiled<false>(record, source, result);
    if (executed) {
      return;
    }
  }
  ExecuteGeneric(record, source, result);
}

template <bool kUnitScale, typename Element, typename Scale>
void ExecuteRecordRangeImpl(const Record& record, const Element* source,
                            Element* result, bool compact,
                            uint64_t unit_start, uint64_t unit_count,
                            Scale scale) {
  if (unit_count == 0) {
    return;
  }
  if (compact) {
    const Element* input =
        source + record.source_offset + unit_start;
    Element* output =
        result + record.destination_offset + unit_start;
    if constexpr (kUnitScale) {
      std::memcpy(output, input,
                  static_cast<std::size_t>(unit_count) * sizeof(Element));
    } else {
      for (uint64_t element = 0; element < unit_count; ++element) {
        output[element] = scale(input[element], record);
      }
    }
    return;
  }

  if (record.rank == 0) {
    if constexpr (kUnitScale) {
      result[record.destination_offset] = source[record.source_offset];
    } else {
      result[record.destination_offset] =
          scale(source[record.source_offset], record);
    }
    return;
  }

  ExecuteGenericRecordRangeImpl<kUnitScale>(
      record, source, result, unit_start, unit_count, scale);
}

void ExecuteRecordRange(const Record& record, const float* source,
                        float* result, RecordWorkKind kind,
                        uint64_t unit_start, uint64_t unit_count) {
  if (kind == RecordWorkKind::kRank2SignedPermutation) {
    if (record.scale == 1.0F) {
      ExecuteRank2SignedPermutationRange<true>(
          record, source, result, unit_start, unit_count);
    } else {
      ExecuteRank2SignedPermutationRange<false>(
          record, source, result, unit_start, unit_count);
    }
    return;
  }
  if (kind == RecordWorkKind::kRank2Forward) {
    ExecuteRank2TransposeColumns(record, source, result, unit_start,
                                 unit_count);
    return;
  }
  if (kind == RecordWorkKind::kRank2Reverse) {
    ExecuteRank2TransposeRows(record, source, result, unit_start,
                              unit_count);
    return;
  }
  if (kind == RecordWorkKind::kRank4Axis0) {
    Record slice = record;
    slice.source_offset += unit_start * record.source_strides[0];
    slice.destination_offset +=
        unit_start * record.destination_strides[0];
    slice.shape[0] = unit_count;
    slice.logical_elements =
        unit_count * (record.logical_elements / record.shape[0]);
    ExecuteRecord(slice, source, result);
    return;
  }
  if (kind == RecordWorkKind::kRank4TwoPair) {
    const bool executed =
        record.scale == 1.0F
            ? ExecuteRank4TwoPair<true>(record, source, result, unit_start,
                                        unit_count)
            : ExecuteRank4TwoPair<false>(record, source, result, unit_start,
                                         unit_count);
    if (!executed) {
      Record slice = record;
      slice.source_offset += unit_start * record.source_strides[0];
      slice.destination_offset +=
          unit_start * record.destination_strides[0];
      slice.shape[0] = unit_count;
      slice.logical_elements =
          unit_count * (record.logical_elements / record.shape[0]);
      ExecuteGeneric(slice, source, result);
    }
    return;
  }
  const auto scale = [](float value, const Record& item) {
    return item.scale * value;
  };
  const bool compact = kind == RecordWorkKind::kCompact;
  if (record.scale == 1.0F) {
    ExecuteRecordRangeImpl<true>(record, source, result, compact, unit_start,
                                 unit_count, scale);
  } else {
    ExecuteRecordRangeImpl<false>(record, source, result, compact, unit_start,
                                  unit_count, scale);
  }
}

template <typename To, typename From>
To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
  uint32_t exponent = (value >> 10) & 0x1FU;
  uint32_t mantissa = value & 0x03FFU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      uint32_t float_exponent = 113;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1;
        --float_exponent;
      }
      mantissa &= 0x03FFU;
      bits = sign | (float_exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    exponent += 112;
    bits = sign | (exponent << 23) | (mantissa << 13);
  }
  return BitCast<float>(bits);
}

uint16_t FloatToHalf(float value) {
  const uint32_t bits = BitCast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
  const uint32_t exponent = (bits >> 23) & 0xFFU;
  const uint32_t mantissa = bits & 0x007FFFFFU;
  if (exponent == 0xFFU) {
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7C00U);
    }
    uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
    if (payload == 0) {
      payload = 1;
    }
    return static_cast<uint16_t>(sign | 0x7C00U | payload);
  }

  int32_t half_exponent = static_cast<int32_t>(exponent) - 112;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const uint32_t significand = mantissa | 0x00800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t rounded = significand >> shift;
    const uint32_t remainder = significand & ((UINT32_C(1) << shift) - 1);
    const uint32_t halfway = UINT32_C(1) << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_mantissa & 1U) != 0)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x0400U) {
      rounded_mantissa = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
      }
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint16_t>(half_exponent) << 10) |
      static_cast<uint16_t>(rounded_mantissa));
}

float BFloat16ToFloat(uint16_t value) {
  return BitCast<float>(static_cast<uint32_t>(value) << 16);
}

uint16_t FloatToBFloat16(float value) {
  uint32_t bits = BitCast<uint32_t>(value);
  if ((bits & 0x7F800000U) == 0x7F800000U &&
      (bits & 0x007FFFFFU) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040U);
  }
  bits += 0x00007FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

uint16_t ScaleF16(uint16_t value, const Record& record) {
  return FloatToHalf(HalfToFloat(value) *
                     HalfToFloat(
                         static_cast<uint16_t>(record.scale_real_bits)));
}

uint16_t ScaleBF16(uint16_t value, const Record& record) {
  return FloatToBFloat16(
      BFloat16ToFloat(value) *
      BFloat16ToFloat(static_cast<uint16_t>(record.scale_real_bits)));
}

float RealPartMultiplyC64(float left_real, float left_imaginary,
                          float right_real, float right_imaginary) {
  return std::fma(left_real, right_real,
                  -(left_imaginary * right_imaginary));
}

std::complex<float> MultiplyC64(float left_real, float left_imaginary,
                                float right_real, float right_imaginary) {
  return {RealPartMultiplyC64(left_real, left_imaginary, right_real,
                              right_imaginary),
          std::fma(left_imaginary, right_real,
                   left_real * right_imaginary)};
}

std::complex<float> ScaleC64(std::complex<float> value,
                             const Record& record) {
  return MultiplyC64(
      BitCast<float>(static_cast<uint32_t>(record.scale_real_bits)),
      BitCast<float>(static_cast<uint32_t>(record.scale_imaginary_bits)),
      value.real(), value.imag());
}

double RealPartMultiplyC128(double left_real, double left_imaginary,
                            double right_real, double right_imaginary) {
  return std::fma(left_real, right_real,
                  -(left_imaginary * right_imaginary));
}

std::complex<double> MultiplyC128(double left_real, double left_imaginary,
                                  double right_real,
                                  double right_imaginary) {
  return {RealPartMultiplyC128(left_real, left_imaginary, right_real,
                               right_imaginary),
          std::fma(left_imaginary, right_real,
                   left_real * right_imaginary)};
}

double ScaleF64(double value, const Record& record) {
  return BitCast<double>(record.scale_real_bits) * value;
}

std::complex<double> ScaleC128(std::complex<double> value,
                               const Record& record) {
  return MultiplyC128(BitCast<double>(record.scale_real_bits),
                      BitCast<double>(record.scale_imaginary_bits),
                      value.real(), value.imag());
}

bool ScalePred(bool value, const Record& record) {
  return value && record.scale_real_bits != 0;
}

template <typename Element>
Element ScaleInteger(Element value, const Record& record) {
  using Unsigned = std::make_unsigned_t<Element>;
  const Unsigned input = BitCast<Unsigned>(value);
  const Unsigned scale = static_cast<Unsigned>(record.scale_real_bits);
  const Unsigned product = static_cast<Unsigned>(
      static_cast<uint64_t>(input) * static_cast<uint64_t>(scale));
  return BitCast<Element>(product);
}

template <typename Element>
Element AddInteger(Element left, Element right) {
  using Unsigned = std::make_unsigned_t<Element>;
  const Unsigned sum = static_cast<Unsigned>(
      static_cast<uint64_t>(BitCast<Unsigned>(left)) +
      static_cast<uint64_t>(BitCast<Unsigned>(right)));
  return BitCast<Element>(sum);
}

bool IsUnitScale(const Record& record, uint64_t dtype) {
  uint64_t unit_real = 1;
  if (dtype == kDtypeF16) {
    unit_real = 0x3C00U;
  } else if (dtype == kDtypeBF16) {
    unit_real = 0x3F80U;
  } else if (dtype == kDtypeF32 || dtype == kDtypeC64) {
    unit_real = 0x3F800000U;
  } else if (dtype == kDtypeF64 || dtype == kDtypeC128) {
    unit_real = UINT64_C(0x3FF0000000000000);
  }
  return record.scale_real_bits == unit_real &&
         record.scale_imaginary_bits == 0;
}

template <typename Element, typename Scale>
void ExecuteScalarRecord(const Record& record, const Element* source,
                         Element* result, bool unit_scale, Scale scale);

template <typename Element, typename Scale>
void ExecuteScalarRecordRange(const Record& record, const Element* source,
                              Element* result, bool unit_scale, Scale scale,
                              bool compact, uint64_t unit_start,
                              uint64_t unit_count);

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_FMA_TARGET
__m256 ScaleFourC64Avx2(__m256 value, float scale_real,
                        float scale_imaginary) {
  const __m256 first_scale = _mm256_set_ps(
      scale_imaginary, scale_real, scale_imaginary, scale_real,
      scale_imaginary, scale_real, scale_imaginary, scale_real);
  const __m256 second_scale = _mm256_set_ps(
      scale_real, scale_imaginary, scale_real, scale_imaginary,
      scale_real, scale_imaginary, scale_real, scale_imaginary);
  const __m256 real = _mm256_moveldup_ps(value);
  const __m256 imaginary = _mm256_movehdup_ps(value);
  __m256 base = _mm256_mul_ps(imaginary, second_scale);
  const __m256 negate_real = _mm256_castsi256_ps(
      _mm256_set1_epi64x(INT64_C(0x0000000080000000)));
  base = _mm256_xor_ps(base, negate_real);
  return _mm256_fmadd_ps(real, first_scale, base);
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
bool HasNonFiniteC64LaneAvx2(__m256 input, __m256 output) {
  const __m256i magnitude_mask = _mm256_set1_epi32(0x7FFFFFFF);
  const __m256i maximum_finite = _mm256_set1_epi32(0x7F7FFFFF);
  const __m256i input_magnitude = _mm256_and_si256(
      _mm256_castps_si256(input), magnitude_mask);
  const __m256i output_magnitude = _mm256_and_si256(
      _mm256_castps_si256(output), magnitude_mask);
  const __m256i invalid = _mm256_or_si256(
      _mm256_cmpgt_epi32(input_magnitude, maximum_finite),
      _mm256_cmpgt_epi32(output_magnitude, maximum_finite));
  return _mm256_movemask_ps(_mm256_castsi256_ps(invalid)) != 0;
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
__m256 MultiplyFourC64ExactAvx2(__m256 value,
                                const std::complex<float>* input,
                                float scale_real,
                                float scale_imaginary) {
  __m256 mapped = ScaleFourC64Avx2(value, scale_real, scale_imaginary);
  if (HasNonFiniteC64LaneAvx2(value, mapped)) {
    alignas(32) std::complex<float> exact[4];
    for (uint64_t lane = 0; lane < 4; ++lane) {
      exact[lane] = MultiplyC64(scale_real, scale_imaginary,
                                input[lane].real(), input[lane].imag());
    }
    mapped = _mm256_load_ps(reinterpret_cast<const float*>(exact));
  }
  return mapped;
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
std::complex<float> MultiplyOneC64ExactAvx2(std::complex<float> value,
                                            float scale_real,
                                            float scale_imaginary) {
  alignas(32) std::complex<float> input[4] = {value, value, value, value};
  const __m256 packed =
      _mm256_load_ps(reinterpret_cast<const float*>(input));
  const __m256 mapped = MultiplyFourC64ExactAvx2(
      packed, input, scale_real, scale_imaginary);
  _mm256_store_ps(reinterpret_cast<float*>(input), mapped);
  return input[0];
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void TransposeFourC64Avx2(__m256* values) {
  const __m256d row0 = _mm256_castps_pd(values[0]);
  const __m256d row1 = _mm256_castps_pd(values[1]);
  const __m256d row2 = _mm256_castps_pd(values[2]);
  const __m256d row3 = _mm256_castps_pd(values[3]);
  const __m256d low01 = _mm256_unpacklo_pd(row0, row1);
  const __m256d high01 = _mm256_unpackhi_pd(row0, row1);
  const __m256d low23 = _mm256_unpacklo_pd(row2, row3);
  const __m256d high23 = _mm256_unpackhi_pd(row2, row3);
  values[0] = _mm256_castpd_ps(
      _mm256_permute2f128_pd(low01, low23, 0x20));
  values[1] = _mm256_castpd_ps(
      _mm256_permute2f128_pd(high01, high23, 0x20));
  values[2] = _mm256_castpd_ps(
      _mm256_permute2f128_pd(low01, low23, 0x31));
  values[3] = _mm256_castpd_ps(
      _mm256_permute2f128_pd(high01, high23, 0x31));
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteCompactScaledC64Avx2Range(
    const Record& record, const std::complex<float>* source,
    std::complex<float>* result, uint64_t element_start,
    uint64_t element_count) {
  const auto* input = source + record.source_offset + element_start;
  auto* output = result + record.destination_offset + element_start;
  const float scale_real =
      BitCast<float>(static_cast<uint32_t>(record.scale_real_bits));
  const float scale_imaginary =
      BitCast<float>(static_cast<uint32_t>(record.scale_imaginary_bits));
  uint64_t element = 0;
  for (; element + 4 <= element_count; element += 4) {
    const __m256 value = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element));
    const __m256 mapped = MultiplyFourC64ExactAvx2(
        value, input + element, scale_real, scale_imaginary);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element), mapped);
  }
  for (; element < element_count; ++element) {
    output[element] =
        MultiplyOneC64ExactAvx2(input[element], scale_real, scale_imaginary);
  }
}

template <bool kUnitScale>
TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteRank2C64Avx2Range(
    const Record& record, const std::complex<float>* source,
    std::complex<float>* result, RecordWorkKind kind, uint64_t unit_start,
    uint64_t unit_count) {
  constexpr uint64_t kTile = 16;
  const std::size_t outer_axis =
      kind == RecordWorkKind::kRank2Forward ? 1 : 0;
  const std::size_t inner_axis = 1 - outer_axis;
  const uint64_t outer_stop = unit_start + unit_count;
  const uint64_t inner_extent = record.shape[inner_axis];
  const float scale_real =
      BitCast<float>(static_cast<uint32_t>(record.scale_real_bits));
  const float scale_imaginary =
      BitCast<float>(static_cast<uint32_t>(record.scale_imaginary_bits));
  for (uint64_t outer_start = unit_start; outer_start < outer_stop;
       outer_start += kTile) {
    const uint64_t outer_tile_stop =
        std::min(outer_stop, outer_start + kTile);
    for (uint64_t inner_start = 0; inner_start < inner_extent;
         inner_start += kTile) {
      const uint64_t inner_stop =
          std::min(inner_extent, inner_start + kTile);
      uint64_t outer = outer_start;
      for (; outer + 4 <= outer_tile_stop; outer += 4) {
        uint64_t inner = inner_start;
        for (; inner + 4 <= inner_stop; inner += 4) {
          __m256 values[4];
          for (uint64_t outer_lane = 0; outer_lane < 4; ++outer_lane) {
            const int64_t source_address =
                record.source_offset +
                static_cast<int64_t>(outer + outer_lane) *
                    record.source_strides[outer_axis] +
                static_cast<int64_t>(inner) *
                    record.source_strides[inner_axis];
            const auto* input = source + source_address;
            const __m256 value = _mm256_loadu_ps(
                reinterpret_cast<const float*>(input));
            if constexpr (kUnitScale) {
              values[outer_lane] = value;
            } else {
              values[outer_lane] = MultiplyFourC64ExactAvx2(
                  value, input, scale_real, scale_imaginary);
            }
          }
          TransposeFourC64Avx2(values);
          for (uint64_t inner_lane = 0; inner_lane < 4; ++inner_lane) {
            const int64_t destination_address =
                record.destination_offset +
                static_cast<int64_t>(outer) *
                    record.destination_strides[outer_axis] +
                static_cast<int64_t>(inner + inner_lane) *
                    record.destination_strides[inner_axis];
            _mm256_storeu_ps(
                reinterpret_cast<float*>(result + destination_address),
                values[inner_lane]);
          }
        }
        for (; inner < inner_stop; ++inner) {
          for (uint64_t outer_lane = 0; outer_lane < 4; ++outer_lane) {
            const int64_t source_address =
                record.source_offset +
                static_cast<int64_t>(outer + outer_lane) *
                    record.source_strides[outer_axis] +
                static_cast<int64_t>(inner) *
                    record.source_strides[inner_axis];
            const int64_t destination_address =
                record.destination_offset +
                static_cast<int64_t>(outer + outer_lane) *
                    record.destination_strides[outer_axis] +
                static_cast<int64_t>(inner) *
                    record.destination_strides[inner_axis];
            if constexpr (kUnitScale) {
              result[destination_address] = source[source_address];
            } else {
              result[destination_address] =
                  MultiplyOneC64ExactAvx2(source[source_address], scale_real,
                                          scale_imaginary);
            }
          }
        }
      }
      for (; outer < outer_tile_stop; ++outer) {
        for (uint64_t inner = inner_start; inner < inner_stop; ++inner) {
          const int64_t source_address =
              record.source_offset +
              static_cast<int64_t>(outer) *
                  record.source_strides[outer_axis] +
              static_cast<int64_t>(inner) *
                  record.source_strides[inner_axis];
          const int64_t destination_address =
              record.destination_offset +
              static_cast<int64_t>(outer) *
                  record.destination_strides[outer_axis] +
              static_cast<int64_t>(inner) *
                  record.destination_strides[inner_axis];
          if constexpr (kUnitScale) {
            result[destination_address] = source[source_address];
          } else {
            result[destination_address] =
                MultiplyOneC64ExactAvx2(source[source_address], scale_real,
                                        scale_imaginary);
          }
        }
      }
    }
  }
}
#endif

template <bool kUnitScale>
void ExecuteRank2C64PortableRange(
    const Record& record, const std::complex<float>* source,
    std::complex<float>* result, RecordWorkKind kind, uint64_t unit_start,
    uint64_t unit_count) {
  constexpr uint64_t kTile = 16;
  const std::size_t outer_axis =
      kind == RecordWorkKind::kRank2Forward ? 1 : 0;
  const std::size_t inner_axis = 1 - outer_axis;
  const uint64_t outer_stop = unit_start + unit_count;
  const uint64_t inner_extent = record.shape[inner_axis];
  for (uint64_t outer_start = unit_start; outer_start < outer_stop;
       outer_start += kTile) {
    const uint64_t outer_tile_stop =
        std::min(outer_stop, outer_start + kTile);
    for (uint64_t inner_start = 0; inner_start < inner_extent;
         inner_start += kTile) {
      const uint64_t inner_stop =
          std::min(inner_extent, inner_start + kTile);
      for (uint64_t outer = outer_start; outer < outer_tile_stop; ++outer) {
        int64_t source_address =
            record.source_offset +
            static_cast<int64_t>(outer) * record.source_strides[outer_axis] +
            static_cast<int64_t>(inner_start) *
                record.source_strides[inner_axis];
        int64_t destination_address =
            record.destination_offset +
            static_cast<int64_t>(outer) *
                record.destination_strides[outer_axis] +
            static_cast<int64_t>(inner_start) *
                record.destination_strides[inner_axis];
        for (uint64_t inner = inner_start; inner < inner_stop; ++inner) {
          if constexpr (kUnitScale) {
            result[destination_address] = source[source_address];
          } else {
            result[destination_address] =
                ScaleC64(source[source_address], record);
          }
          source_address += record.source_strides[inner_axis];
          destination_address += record.destination_strides[inner_axis];
        }
      }
    }
  }
}

void ExecuteCompactC64Range(const Record& record,
                            const std::complex<float>* source,
                            std::complex<float>* result, bool unit_scale,
                            uint64_t element_start,
                            uint64_t element_count) {
  if (unit_scale) {
    std::memcpy(result + record.destination_offset + element_start,
                source + record.source_offset + element_start,
                static_cast<std::size_t>(element_count) *
                    sizeof(std::complex<float>));
    return;
  }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2Fma()) {
    ExecuteCompactScaledC64Avx2Range(record, source, result, element_start,
                                     element_count);
    return;
  }
#endif
  for (uint64_t element = 0; element < element_count; ++element) {
    const uint64_t index = element_start + element;
    result[record.destination_offset + index] =
        ScaleC64(source[record.source_offset + index], record);
  }
}

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteDynamicScaledC64Avx2Range(
    const std::complex<float>* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
  const float scale_real = factor.real();
  const float scale_imaginary = factor.imag();
  uint64_t element = 0;
  for (; element + 4 <= element_count; element += 4) {
    const __m256 value = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element));
    const __m256 mapped = MultiplyFourC64ExactAvx2(
        value, input + element, scale_real, scale_imaginary);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element), mapped);
  }
  for (; element < element_count; ++element) {
    output[element] = MultiplyOneC64ExactAvx2(
        input[element], scale_real, scale_imaginary);
  }
}
#endif

void ExecuteDynamicScaledC64Range(
    const std::complex<float>* input, std::complex<float>* output,
    uint64_t element_count, std::complex<float> factor) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2Fma()) {
    ExecuteDynamicScaledC64Avx2Range(input, output, element_count, factor);
    return;
  }
#endif
  for (uint64_t element = 0; element < element_count; ++element) {
    output[element] =
        MultiplyC64(factor.real(), factor.imag(), input[element].real(),
                    input[element].imag());
  }
}

void ExecuteRank2C64Range(const Record& record,
                          const std::complex<float>* source,
                          std::complex<float>* result, RecordWorkKind kind,
                          bool unit_scale, uint64_t unit_start,
                          uint64_t unit_count) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2Fma()) {
    if (unit_scale) {
      ExecuteRank2C64Avx2Range<true>(record, source, result, kind, unit_start,
                                     unit_count);
    } else {
      ExecuteRank2C64Avx2Range<false>(record, source, result, kind, unit_start,
                                      unit_count);
    }
    return;
  }
#endif
  if (unit_scale) {
    ExecuteRank2C64PortableRange<true>(record, source, result, kind,
                                       unit_start, unit_count);
  } else {
    ExecuteRank2C64PortableRange<false>(record, source, result, kind,
                                        unit_start, unit_count);
  }
}

void ExecuteC64Record(const Record& record,
                      const std::complex<float>* source,
                      std::complex<float>* result) {
  const bool unit_scale = record.scale_real_bits == 0x3F800000U &&
                          record.scale_imaginary_bits == 0;
  if (record.logical_elements == 0) {
    return;
  }
  if (IsCompactSameMapping(record)) {
    ExecuteCompactC64Range(record, source, result, unit_scale, 0,
                           record.logical_elements);
    return;
  }
  if (IsRank2ForwardCandidate(record)) {
    ExecuteRank2C64Range(record, source, result,
                         RecordWorkKind::kRank2Forward, unit_scale, 0,
                         record.shape[1]);
    return;
  }
  if (IsRank2ReverseCandidate(record)) {
    ExecuteRank2C64Range(record, source, result,
                         RecordWorkKind::kRank2Reverse, unit_scale, 0,
                         record.shape[0]);
    return;
  }
  ExecuteScalarRecord(record, source, result, unit_scale, ScaleC64);
}

void ExecuteC64RecordRange(const Record& record,
                           const std::complex<float>* source,
                           std::complex<float>* result, RecordWorkKind kind,
                           uint64_t unit_start, uint64_t unit_count) {
  const bool unit_scale = record.scale_real_bits == 0x3F800000U &&
                          record.scale_imaginary_bits == 0;
  if (kind == RecordWorkKind::kCompact) {
    ExecuteCompactC64Range(record, source, result, unit_scale, unit_start,
                           unit_count);
    return;
  }
  if (kind == RecordWorkKind::kRank2Forward ||
      kind == RecordWorkKind::kRank2Reverse) {
    ExecuteRank2C64Range(record, source, result, kind, unit_scale, unit_start,
                         unit_count);
    return;
  }
  ExecuteScalarRecordRange(record, source, result, unit_scale, ScaleC64,
                           false, unit_start, unit_count);
}

int32_t ScaleS32(int32_t value, const Record& record) {
  return ScaleInteger(value, record);
}

template <bool kUnitScale, typename Element, typename Scale>
void ExecuteScalarRecordImpl(const Record& record, const Element* source,
                             Element* result, Scale scale) {
  if (record.logical_elements == 0) {
    return;
  }
  if (IsCompactSameMapping(record)) {
    const Element* input = source + record.source_offset;
    Element* output = result + record.destination_offset;
    if constexpr (kUnitScale) {
      std::memcpy(output, input,
                  static_cast<std::size_t>(record.logical_elements) *
                      sizeof(Element));
    } else {
      for (uint64_t element = 0; element < record.logical_elements;
           ++element) {
        output[element] = scale(input[element], record);
      }
    }
    return;
  }
  if (record.rank == 0) {
    if constexpr (kUnitScale) {
      result[record.destination_offset] = source[record.source_offset];
    } else {
      result[record.destination_offset] =
          scale(source[record.source_offset], record);
    }
    return;
  }

  const uint64_t row_count = GenericRowCount(record);
  ExecuteGenericRecordRangeImpl<kUnitScale>(
      record, source, result, 0, row_count, scale);
}

template <typename Element, typename Scale>
void ExecuteScalarRecord(const Record& record, const Element* source,
                         Element* result, bool unit_scale, Scale scale) {
  if (unit_scale) {
    ExecuteScalarRecordImpl<true>(record, source, result, scale);
  } else {
    ExecuteScalarRecordImpl<false>(record, source, result, scale);
  }
}

template <typename Element, typename Scale>
void ExecuteScalarRecordRange(const Record& record, const Element* source,
                              Element* result, bool unit_scale, Scale scale,
                              bool compact, uint64_t unit_start,
                              uint64_t unit_count) {
  if (unit_scale) {
    ExecuteRecordRangeImpl<true>(record, source, result, compact, unit_start,
                                 unit_count, scale);
  } else {
    ExecuteRecordRangeImpl<false>(record, source, result, compact, unit_start,
                                  unit_count, scale);
  }
}

template <typename SourceElement, typename ResultElement, typename ScalarOp>
void ExecuteAffineRecordRange(const Record& record,
                              const SourceElement* source,
                              ResultElement* result, bool compact,
                              uint64_t unit_start, uint64_t unit_count,
                              ScalarOp scalar_op) {
  if (unit_count == 0) {
    return;
  }
  if (compact) {
    const SourceElement* input =
        source + record.source_offset + unit_start;
    ResultElement* output =
        result + record.destination_offset + unit_start;
    for (uint64_t element = 0; element < unit_count; ++element) {
      output[element] = scalar_op(input[element], record);
    }
    return;
  }
  if (record.rank == 0) {
    result[record.destination_offset] =
        scalar_op(source[record.source_offset], record);
    return;
  }

  ExecuteGenericRecordRangeImpl<false>(
      record, source, result, unit_start, unit_count, scalar_op);
}

template <typename SourceElement, typename ResultElement, typename ScalarOp>
void ExecuteAffineRecord(const Record& record, const SourceElement* source,
                         ResultElement* result, ScalarOp scalar_op) {
  const bool compact = IsCompactSameMapping(record);
  const uint64_t unit_count = compact || record.rank == 0
                                  ? record.logical_elements
                                  : GenericRowCount(record);
  ExecuteAffineRecordRange(record, source, result, compact, 0, unit_count,
                           scalar_op);
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteGenericUpdateRecordRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t row_start, uint64_t row_count, ScalarOp scalar_op,
    WriteOp write_op) {
  if (row_count == 0) {
    return;
  }
  const std::size_t fastest_axis = record.loop_axes_fastest_first[0];
  const uint64_t inner_elements = record.shape[fastest_axis];
  const int64_t source_inner_stride = record.source_strides[fastest_axis];
  const int64_t destination_inner_stride =
      record.destination_strides[fastest_axis];
  std::array<uint64_t, kMaximumRank> coordinates{};
  int64_t source_base = record.source_offset;
  int64_t destination_base = record.destination_offset;
  uint64_t remaining = row_start;
  for (std::size_t position = 1; position < record.rank; ++position) {
    const std::size_t axis = record.loop_axes_fastest_first[position];
    coordinates[axis] = remaining % record.shape[axis];
    remaining /= record.shape[axis];
    source_base += static_cast<int64_t>(coordinates[axis]) *
                   record.source_strides[axis];
    destination_base += static_cast<int64_t>(coordinates[axis]) *
                        record.destination_strides[axis];
  }

  for (uint64_t row = 0; row < row_count; ++row) {
    int64_t source_index = source_base;
    int64_t destination_index = destination_base;
    for (uint64_t inner = 0; inner < inner_elements; ++inner) {
      write_op(result[destination_index],
               scalar_op(source[source_index], record));
      source_index += source_inner_stride;
      destination_index += destination_inner_stride;
    }
    if (row + 1 == row_count) {
      break;
    }
    for (std::size_t position = 1; position < record.rank; ++position) {
      const std::size_t axis = record.loop_axes_fastest_first[position];
      if (coordinates[axis] + 1 < record.shape[axis]) {
        ++coordinates[axis];
        source_base += record.source_strides[axis];
        destination_base += record.destination_strides[axis];
        break;
      }
      source_base -= static_cast<int64_t>(coordinates[axis]) *
                     record.source_strides[axis];
      destination_base -= static_cast<int64_t>(coordinates[axis]) *
                          record.destination_strides[axis];
      coordinates[axis] = 0;
    }
  }
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteAffineUpdateRecord(const Record& record,
                               const SourceElement* source,
                               ResultElement* result, ScalarOp scalar_op,
                               WriteOp write_op) {
  if (record.logical_elements == 0) {
    return;
  }
  if (record.rank == 0) {
    write_op(result[record.destination_offset],
             scalar_op(source[record.source_offset], record));
    return;
  }
  ExecuteGenericUpdateRecordRange(record, source, result, 0,
                                  GenericRowCount(record), scalar_op,
                                  write_op);
}

Record SliceRecordWorkRange(const Record& record, RecordWorkKind kind,
                            uint64_t unit_start, uint64_t unit_count) {
  Record slice = record;
  std::size_t axis = 0;
  if (kind == RecordWorkKind::kRank2Forward) {
    axis = 1;
  } else if (kind == RecordWorkKind::kRank2Reverse ||
             kind == RecordWorkKind::kRank4Axis0 ||
             kind == RecordWorkKind::kRank4TwoPair) {
    axis = 0;
  } else if (kind == RecordWorkKind::kRank2SignedPermutation) {
    axis = record.shape[0] >= record.shape[1] ? 1 : 0;
  }
  slice.source_offset +=
      static_cast<int64_t>(unit_start) * record.source_strides[axis];
  slice.destination_offset +=
      static_cast<int64_t>(unit_start) * record.destination_strides[axis];
  slice.shape[axis] = unit_count;
  slice.logical_elements =
      unit_count * (record.logical_elements / record.shape[axis]);
  return slice;
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteAffineUpdateRecordRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    RecordWorkKind kind, uint64_t unit_start, uint64_t unit_count,
    ScalarOp scalar_op, WriteOp write_op) {
  if (unit_count == 0) {
    return;
  }
  if (kind == RecordWorkKind::kCompact) {
    const SourceElement* input =
        source + record.source_offset + unit_start;
    ResultElement* output =
        result + record.destination_offset + unit_start;
    for (uint64_t element = 0; element < unit_count; ++element) {
      write_op(output[element], scalar_op(input[element], record));
    }
    return;
  }
  if (kind == RecordWorkKind::kGeneric) {
    ExecuteGenericUpdateRecordRange(record, source, result, unit_start,
                                    unit_count, scalar_op, write_op);
    return;
  }
  ExecuteAffineUpdateRecord(
      SliceRecordWorkRange(record, kind, unit_start, unit_count), source,
      result, scalar_op, write_op);
}

struct RecordWork {
  std::size_t record_index = 0;
  uint64_t chunk_begin = 0;
  uint64_t chunk_count = 0;
  uint64_t units_per_chunk = 0;
  uint64_t unit_count = 0;
  uint64_t elements_per_unit = 0;
  RecordWorkKind kind = RecordWorkKind::kGeneric;
};

constexpr uint64_t kParallelChunkBytes = UINT64_C(256) * 1024;
// Fresh-process crossover pilots on the locked production matrix:
// - general rank-2 and tiled rank-4 copies gained 26% and 37% at ~0.75 MiB;
// - compact copies gained only 8-10% at 2 MiB, but over 2x at 4 MiB.
// Keep the compact threshold conservative because it is bandwidth-bound and
// XLA-pool scheduling competes with concurrent compiled calls.
constexpr uint64_t kParallelGeneralMinimumBytes = UINT64_C(512) * 1024;
constexpr uint64_t kParallelCompactMinimumBytes = UINT64_C(4) * 1024 * 1024;
constexpr uint64_t kParallelRank4MinimumBytes = UINT64_C(512) * 1024;

bool IsRank4TiledCandidate(const Record& record) {
  if (record.rank != 4) {
    return false;
  }
  const bool forward =
      record.source_strides[1] == 1 &&
      record.destination_strides[3] == 1;
  const bool reverse =
      record.source_strides[3] == 1 &&
      record.destination_strides[1] == 1;
  return forward || reverse;
}

bool IsRank4TwoPairCandidate(const Record& record) {
  if (record.rank != 4 || record.shape[2] % 8 != 0 ||
      record.shape[3] % 8 != 0) {
    return false;
  }
  uint64_t inner = 0;
  uint64_t source_axis1 = 0;
  uint64_t destination_axis0 = 0;
  uint64_t destination_axis1_stride = 0;
  return CheckedMultiply(record.shape[2], record.shape[3], &inner) &&
         CheckedMultiply(record.shape[0], inner, &source_axis1) &&
         PositiveStrideValue(record.destination_strides[1],
                             &destination_axis1_stride) &&
         CheckedMultiply(record.shape[1], destination_axis1_stride,
                         &destination_axis0) &&
         StrideEquals(record.source_strides[0], inner) &&
         StrideEquals(record.source_strides[1], source_axis1) &&
         record.source_strides[2] == 1 &&
         StrideEquals(record.source_strides[3], record.shape[2]) &&
         record.destination_strides[3] == 1 &&
         StrideEquals(record.destination_strides[2], record.shape[3]) &&
         destination_axis1_stride >= inner &&
         StrideEquals(record.destination_strides[0], destination_axis0);
}

uint64_t ParallelMinimumBytes(const ParsedPlan& plan) {
  bool all_compact = true;
  bool has_nonunit_c64_scale = false;
  bool has_rank4_tiled = false;
  for (const Record& record : plan.records) {
    const bool compact = IsCompactSameMapping(record);
    all_compact = all_compact && compact;
    has_nonunit_c64_scale =
        has_nonunit_c64_scale ||
        record.scale_real_bits != 0x3F800000U ||
        record.scale_imaginary_bits != 0;
    has_rank4_tiled =
        has_rank4_tiled || (!compact && IsRank4TiledCandidate(record));
  }
  if (plan.dtype == kDtypeC64 && all_compact &&
      has_nonunit_c64_scale) {
    return std::numeric_limits<uint64_t>::max();
  }
  if (all_compact) {
    return kParallelCompactMinimumBytes;
  }
  if (has_rank4_tiled && plan.records.size() == 1) {
    return kParallelRank4MinimumBytes;
  }
  return kParallelGeneralMinimumBytes;
}

ffi::Error BuildRecordWork(const ParsedPlan& plan, uint64_t item_size,
                           std::vector<RecordWork>* work,
                           uint64_t* chunks_per_batch) {
  const uint64_t target_elements =
      std::max<uint64_t>(1, kParallelChunkBytes / item_size);
  uint64_t chunk_cursor = 0;
  work->reserve(plan.records.size());
  for (std::size_t record_index = 0;
       record_index < plan.records.size(); ++record_index) {
    const Record& record = plan.records[record_index];
    if (record.logical_elements == 0) {
      continue;
    }
    RecordWork item;
    item.record_index = record_index;
    item.chunk_begin = chunk_cursor;
    uint64_t elements_per_unit = 1;
    if (IsCompactSameMapping(record)) {
      item.kind = RecordWorkKind::kCompact;
      item.unit_count = record.logical_elements;
    } else if (plan.dtype == kDtypeF32 &&
               IsRank2SignedPermutationCandidate(record)) {
      item.kind = RecordWorkKind::kRank2SignedPermutation;
      if (record.shape[0] >= record.shape[1]) {
        item.unit_count = record.shape[1];
        elements_per_unit = record.shape[0];
      } else {
        item.unit_count = record.shape[0];
        elements_per_unit = record.shape[1];
      }
    } else if ((plan.dtype == kDtypeF32 || plan.dtype == kDtypeC64) &&
               IsRank2ForwardCandidate(record)) {
      item.kind = RecordWorkKind::kRank2Forward;
      item.unit_count = record.shape[1];
      elements_per_unit = record.shape[0];
    } else if ((plan.dtype == kDtypeF32 || plan.dtype == kDtypeC64) &&
               IsRank2ReverseCandidate(record)) {
      item.kind = RecordWorkKind::kRank2Reverse;
      item.unit_count = record.shape[0];
      elements_per_unit = record.shape[1];
    } else if (plan.dtype == kDtypeF32 &&
               IsRank4TiledCandidate(record) &&
               record.shape[0] != 0) {
      item.kind = RecordWorkKind::kRank4Axis0;
      item.unit_count = record.shape[0];
      elements_per_unit =
          record.logical_elements / record.shape[0];
    } else if (plan.dtype == kDtypeF32 &&
               IsRank4TwoPairCandidate(record) &&
               record.shape[0] != 0) {
      item.kind = RecordWorkKind::kRank4TwoPair;
      item.unit_count = record.shape[0];
      elements_per_unit =
          record.logical_elements / record.shape[0];
    } else if (record.rank == 0) {
      item.unit_count = record.logical_elements;
    } else {
      elements_per_unit =
          record.shape[record.loop_axes_fastest_first[0]];
      item.unit_count = GenericRowCount(record);
    }
    item.units_per_chunk =
        std::max<uint64_t>(1, target_elements / elements_per_unit);
    item.elements_per_unit = elements_per_unit;
    item.chunk_count =
        1 + (item.unit_count - 1) / item.units_per_chunk;
    if (!CheckedAdd(chunk_cursor, item.chunk_count, &chunk_cursor)) {
      return Invalid("parallel chunk count overflow");
    }
    work->push_back(item);
  }
  *chunks_per_batch = chunk_cursor;
  return ffi::Error::Success();
}

ffi::Error ExpectCompiledWord(const DescriptorReader& reader,
                              std::size_t* cursor, uint64_t expected,
                              const char* field) {
  uint64_t actual = 0;
  ffi::Error error = ReadWord(reader, cursor, &actual);
  if (!error.success()) {
    return error;
  }
  if (actual != expected) {
    return Invalid(std::string("compiled descriptor ") + field +
                   " mismatch");
  }
  return ffi::Error::Success();
}

bool IsDenseView(const Record& record, bool source) {
  std::array<std::size_t, kMaximumRank> order{};
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    order[axis] = axis;
  }
  const auto& strides = Strides(record, source);
  std::stable_sort(
      order.begin(), order.begin() + record.rank,
      [&](std::size_t left, std::size_t right) {
        return AbsoluteStride(strides[left]) <
               AbsoluteStride(strides[right]);
      });
  uint64_t expected = 1;
  for (std::size_t position = 0; position < record.rank; ++position) {
    const std::size_t axis = order[position];
    if (record.shape[axis] <= 1) {
      continue;
    }
    if (AbsoluteStride(strides[axis]) != expected ||
        !CheckedMultiply(expected, record.shape[axis], &expected)) {
      return false;
    }
  }
  return true;
}

uint64_t CompiledLayoutCode(const Record& record) {
  if (record.logical_elements == 0) {
    return 0;
  }
  if (record.source_broadcast_axis_mask != 0) {
    return 6;
  }
  if (IsCompactSameMapping(record)) {
    return 1;
  }
  if (IsDenseView(record, true) && IsDenseView(record, false)) {
    return 2;
  }
  bool all_positive = true;
  bool all_nonzero = true;
  for (std::size_t axis = 0; axis < record.rank; ++axis) {
    if (record.shape[axis] > 1 &&
        (record.source_strides[axis] <= 0 ||
         record.destination_strides[axis] <= 0)) {
      all_positive = false;
    }
    if (record.shape[axis] > 1 &&
        (record.source_strides[axis] == 0 ||
         record.destination_strides[axis] == 0)) {
      all_nonzero = false;
    }
  }
  if (all_positive) {
    return 3;
  }
  return all_nonzero ? 4 : 5;
}

uint64_t CompiledScaleCode(const Record& record, uint64_t dtype) {
  uint64_t unit_real = 1;
  uint64_t negative_unit_real = 0;
  bool has_negative_unit = false;
  if (dtype == kDtypeF16) {
    unit_real = 0x3C00U;
    negative_unit_real = 0xBC00U;
    has_negative_unit = true;
  } else if (dtype == kDtypeBF16) {
    unit_real = 0x3F80U;
    negative_unit_real = 0xBF80U;
    has_negative_unit = true;
  } else if (dtype == kDtypeF32 || dtype == kDtypeC64) {
    unit_real = 0x3F800000U;
    negative_unit_real = 0xBF800000U;
    has_negative_unit = true;
  } else if (dtype == kDtypeF64 || dtype == kDtypeC128) {
    unit_real = UINT64_C(0x3FF0000000000000);
    negative_unit_real = UINT64_C(0xBFF0000000000000);
    has_negative_unit = true;
  } else if (dtype == kDtypePred) {
    negative_unit_real = 1U;
    has_negative_unit = true;
  } else if (dtype == kDtypeS8) {
    negative_unit_real = 0xFFU;
    has_negative_unit = true;
  } else if (dtype == kDtypeS16) {
    negative_unit_real = 0xFFFFU;
    has_negative_unit = true;
  } else if (dtype == kDtypeS32) {
    negative_unit_real = 0xFFFFFFFFU;
    has_negative_unit = true;
  } else if (dtype == kDtypeS64) {
    negative_unit_real = std::numeric_limits<uint64_t>::max();
    has_negative_unit = true;
  }
  if (record.scale_real_bits == unit_real &&
      record.scale_imaginary_bits == 0) {
    return 1;
  }
  if (has_negative_unit && record.scale_real_bits == negative_unit_real &&
      record.scale_imaginary_bits == 0) {
    return 2;
  }
  return 3;
}

uint64_t CompiledKernelCode(RecordWorkKind kind) {
  if (kind == RecordWorkKind::kCompact) {
    return 1;
  }
  if (kind == RecordWorkKind::kRank2Forward) {
    return 3;
  }
  if (kind == RecordWorkKind::kRank2Reverse) {
    return 4;
  }
  if (kind == RecordWorkKind::kRank4Axis0) {
    return 5;
  }
  if (kind == RecordWorkKind::kRank4TwoPair) {
    return 6;
  }
  if (kind == RecordWorkKind::kRank2SignedPermutation) {
    return 7;
  }
  return 2;
}

ffi::Error ValidateCompiledDescriptor(
    ffi::Span<const uint8_t> descriptor,
    ParsedPlan* verified_plan = nullptr,
    std::vector<RecordWork>* verified_work = nullptr,
    uint64_t* verified_chunks_per_batch = nullptr,
    uint64_t* verified_parallel_minimum_bytes = nullptr) {
  const DescriptorReader reader(descriptor);
  if (!reader.HasWholeWords()) {
    return Invalid(
        "compiled descriptor byte length is not divisible by eight");
  }
  if (reader.size() < kCompiledDescriptorHeaderWords) {
    return Invalid("compiled descriptor is truncated");
  }

  std::array<uint64_t, kCompiledDescriptorHeaderWords> header{};
  for (std::size_t index = 0; index < header.size(); ++index) {
    if (!reader.Read(index, &header[index])) {
      return Invalid("compiled descriptor header is truncated");
    }
  }
  if (header[0] != kCompiledDescriptorMagic) {
    return Invalid("compiled descriptor magic mismatch");
  }
  if (header[1] != kCompiledDescriptorVersion) {
    return Invalid("compiled descriptor version mismatch");
  }
  if (header[2] != reader.size()) {
    return Invalid("compiled descriptor word count mismatch");
  }
  if (header[3] != kCompilerPolicyVersion ||
      header[4] != kCpuPolicyVersion) {
    return Invalid("compiled descriptor policy version mismatch");
  }
  if (header[7] != kParallelChunkBytes) {
    return Invalid("compiled descriptor preferred chunk size mismatch");
  }
  if (header[10] != kRawDescriptorWitness) {
    return Invalid("compiled descriptor witness kind mismatch");
  }

  const uint64_t raw_word_count_u64 = header[5];
  if (raw_word_count_u64 >
      reader.size() - kCompiledDescriptorHeaderWords) {
    return Invalid("compiled descriptor raw witness length is invalid");
  }
  const std::size_t raw_word_count =
      static_cast<std::size_t>(raw_word_count_u64);
  const std::size_t raw_byte_offset =
      kCompiledDescriptorHeaderWords * sizeof(uint64_t);
  const ffi::Span<const uint8_t> raw_descriptor(
      descriptor.begin() + raw_byte_offset,
      raw_word_count * sizeof(uint64_t));

  ParsedPlan plan;
  ffi::Error error = ParseAndValidateDescriptor(
      DescriptorReader(raw_descriptor), &plan);
  if (!error.success()) {
    return error;
  }
  if (DtypeItemSize(header[11]) == 0 || DtypeItemSize(header[12]) == 0) {
    return Invalid("compiled descriptor operand dtype is invalid");
  }
  if (header[13] != kScalarForwardScaleCast &&
      header[13] != kScalarJaxTranspose) {
    return Invalid("compiled descriptor scalar policy is invalid");
  }
  const uint64_t expected_scalar_dtype =
      header[13] == kScalarJaxTranspose ? header[11] : header[12];
  if (plan.dtype != expected_scalar_dtype) {
    return Invalid("compiled descriptor scalar dtype witness mismatch");
  }
  plan.source_dtype = header[11];
  plan.result_dtype = header[12];
  plan.scalar_kind = header[13];
  if (header[6] != plan.records.size()) {
    return Invalid("compiled descriptor record count mismatch");
  }

  std::vector<AxisProvenance> provenance(plan.records.size());
  for (std::size_t record_index = 0;
       record_index < plan.records.size(); ++record_index) {
    OptimizeRecordForExecution(&plan.records[record_index],
                               &provenance[record_index]);
  }

  const uint64_t item_size = DtypeItemSize(plan.dtype);
  std::vector<RecordWork> work;
  uint64_t chunks_per_batch = 0;
  error = BuildRecordWork(plan, item_size, &work, &chunks_per_batch);
  if (!error.success()) {
    return error;
  }
  if (header[8] != ParallelMinimumBytes(plan)) {
    return Invalid("compiled descriptor parallel threshold mismatch");
  }
  if (header[9] != chunks_per_batch) {
    return Invalid("compiled descriptor chunk count mismatch");
  }

  std::size_t cursor =
      kCompiledDescriptorHeaderWords + raw_word_count;
  std::size_t work_index = 0;
  for (std::size_t record_index = 0;
       record_index < plan.records.size(); ++record_index) {
    const Record& record = plan.records[record_index];
    const bool empty = record.logical_elements == 0;
    const RecordWork* record_work = nullptr;
    if (!empty) {
      if (work_index >= work.size() ||
          work[work_index].record_index != record_index) {
        return Invalid("compiled descriptor work record order mismatch");
      }
      record_work = &work[work_index++];
    }

    const uint64_t kernel_code =
        empty ? 0 : CompiledKernelCode(record_work->kind);
    const uint64_t elements_per_unit =
        empty ? 0 : record_work->elements_per_unit;
    const uint64_t unit_count = empty ? 0 : record_work->unit_count;
    const uint64_t units_per_chunk =
        empty ? 0 : record_work->units_per_chunk;
    const uint64_t chunk_count = empty ? 0 : record_work->chunk_count;
    const uint64_t loop_count = empty ? 0 : record.rank;

    for (const auto& expected : {
             std::pair<uint64_t, const char*>{record.rank, "record rank"},
             {CompiledLayoutCode(record), "record layout kind"},
             {CompiledScaleCode(record, plan.dtype), "record scale kind"},
             {kernel_code, "record kernel kind"},
             {record.logical_elements, "record logical elements"},
             {record.source_offset, "record source offset"},
             {record.destination_offset, "record destination offset"},
             {record.scale_real_bits, "record scale real bits"},
             {record.scale_imaginary_bits,
              "record scale imaginary bits"},
             {record.source_broadcast_axis_mask,
              "record source broadcast mask"},
             {elements_per_unit, "record elements per unit"},
             {unit_count, "record unit count"},
             {units_per_chunk, "record units per chunk"},
             {chunk_count, "record chunk count"},
             {loop_count, "record loop-order length"},
             {provenance[record_index].size(),
              "record provenance group count"},
         }) {
      error = ExpectCompiledWord(reader, &cursor, expected.first,
                                 expected.second);
      if (!error.success()) {
        return error;
      }
    }

    for (std::size_t axis = 0; axis < record.rank; ++axis) {
      error = ExpectCompiledWord(reader, &cursor, record.shape[axis],
                                 "record shape");
      if (!error.success()) {
        return error;
      }
    }
    for (std::size_t axis = 0; axis < record.rank; ++axis) {
      error = ExpectCompiledWord(reader, &cursor,
                                 record.source_strides[axis],
                                 "record source stride");
      if (!error.success()) {
        return error;
      }
    }
    for (std::size_t axis = 0; axis < record.rank; ++axis) {
      error = ExpectCompiledWord(reader, &cursor,
                                 record.destination_strides[axis],
                                 "record destination stride");
      if (!error.success()) {
        return error;
      }
    }
    for (std::size_t position = 0; position < loop_count; ++position) {
      error = ExpectCompiledWord(
          reader, &cursor, record.loop_axes_fastest_first[position],
          "record loop axis");
      if (!error.success()) {
        return error;
      }
    }
    for (const auto& group : provenance[record_index]) {
      error = ExpectCompiledWord(reader, &cursor, group.size(),
                                 "provenance group length");
      if (!error.success()) {
        return error;
      }
      for (std::size_t axis : group) {
        error = ExpectCompiledWord(reader, &cursor, axis,
                                   "provenance axis");
        if (!error.success()) {
          return error;
        }
      }
    }
  }
  if (work_index != work.size()) {
    return Invalid("compiled descriptor work record count mismatch");
  }
  if (cursor != reader.size()) {
    return Invalid("compiled descriptor has trailing words");
  }
  const bool requested_execution = verified_plan != nullptr;
  if (requested_execution != (verified_work != nullptr) ||
      requested_execution != (verified_chunks_per_batch != nullptr) ||
      requested_execution != (verified_parallel_minimum_bytes != nullptr)) {
    return Invalid("compiled execution outputs must be requested together");
  }
  if (requested_execution) {
    *verified_plan = std::move(plan);
    *verified_work = std::move(work);
    *verified_chunks_per_batch = chunks_per_batch;
    *verified_parallel_minimum_bytes = header[8];
  }
  return ffi::Error::Success();
}


ffi::Future ErrorFuture(ffi::Error error) {
  ffi::Promise promise;
  ffi::Future future(promise);
  promise.SetError(std::move(error));
  return future;
}

ffi::Future ReadyFuture() {
  ffi::Promise promise;
  ffi::Future future(promise);
  promise.SetAvailable();
  return future;
}


struct PreparedExecutionData {
  ParsedPlan plan;
  std::vector<RecordWork> work;
  uint64_t chunks_per_batch = 0;
  uint64_t parallel_minimum_bytes = 0;
};

struct ReductionRecord {
  std::size_t rank = 0;
  int64_t input_offset = 0;
  int64_t output_offset = 0;
  uint64_t scale_real_bits = 0;
  uint64_t scale_imaginary_bits = 0;
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
};

uint64_t TrackedReductionPlanBytes(const ParsedReductionPlan& plan) {
  return sizeof(ParsedReductionPlan) +
         plan.records.capacity() * sizeof(ReductionRecord);
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

ffi::Error ValidateReductionInjectiveView(
    const ReductionRecord& record, bool input, uint64_t axis_mask) {
  const auto& strides = input ? record.input_strides : record.output_strides;
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
      return Invalid(std::string(input ? "reduction input" :
                                         "reduction output map") +
                     " does not prove an injective view");
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
    if (reduction_axis && extent > 1 &&
        record->input_strides[axis] == 0) {
      return Invalid("reduction fiber input has a zero nontrivial stride");
    }
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
  uint64_t injective_input_mask = valid_mask;
  for (std::size_t axis = 0; axis < record->rank; ++axis) {
    if (record->shape[axis] > 1 && record->input_strides[axis] == 0) {
      injective_input_mask &= ~(UINT64_C(1) << axis);
    }
  }
  ffi::Error error = ValidateReductionInjectiveView(
      *record, true, injective_input_mask);
  if (!error.success()) {
    return error;
  }
  error = ValidateReductionInjectiveView(
      *record, false, record->map_axis_mask);
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
      header[2] != reader.size() ||
      header[3] != kReductionCompilerPolicyVersion) {
    return Invalid("reduction descriptor header mismatch");
  }
  plan->input_size = header[4];
  plan->output_size = header[5];
  const uint64_t record_count_word = header[6];
  plan->input_dtype = header[7];
  plan->mapped_dtype = header[8];
  plan->output_dtype = header[9];
  plan->scalar_policy = header[10];
  plan->preferred_chunk_bytes = header[11];
  plan->parallel_minimum_bytes = header[12];
  if (record_count_word > reader.size() ||
      record_count_word > std::numeric_limits<std::size_t>::max()) {
    return Invalid("reduction record count exceeds descriptor size");
  }
  const bool forward_scale_cast =
      plan->scalar_policy == kReductionScalarForwardScaleCast;
  const bool jax_transpose =
      plan->scalar_policy == kReductionScalarJaxTranspose;
  if ((!forward_scale_cast && !jax_transpose) ||
      plan->mapped_dtype !=
          (forward_scale_cast ? plan->output_dtype : plan->input_dtype) ||
      !IsReductionDtype(plan->input_dtype) ||
      !IsReductionDtype(plan->output_dtype) ||
      plan->preferred_chunk_bytes == 0) {
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
    ReductionRecord record;
    error = ReadSize(reader, &cursor, &record.rank);
    if (!error.success()) {
      return error;
    }
    if (record.rank > kMaximumRank) {
      return Invalid("reduction logical rank exceeds native limit");
    }
    uint64_t input_offset = 0;
    uint64_t output_offset = 0;
    uint64_t scale_real = 0;
    uint64_t scale_imaginary = 0;
    for (uint64_t* value :
         {&input_offset, &output_offset, &scale_real, &scale_imaginary,
          &record.map_axis_mask, &record.reduction_axis_mask,
          &record.output_count, &record.reduction_count}) {
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
    record.input_offset = static_cast<int64_t>(input_offset);
    record.output_offset = static_cast<int64_t>(output_offset);
    const uint64_t real_limit = ScalarWordLimit(plan->mapped_dtype);
    if (scale_real > real_limit ||
        scale_imaginary > real_limit ||
        (!IsComplexDtype(plan->mapped_dtype) && scale_imaginary != 0)) {
      return Invalid("reduction scale bits do not match mapped dtype");
    }
    record.scale_real_bits = scale_real;
    record.scale_imaginary_bits = scale_imaginary;
    for (std::size_t axis = 0; axis < record.rank; ++axis) {
      error = ReadWord(reader, &cursor, &record.shape[axis]);
      if (!error.success()) {
        return error;
      }
    }
    for (auto* strides : {&record.input_strides, &record.output_strides}) {
      for (std::size_t axis = 0; axis < record.rank; ++axis) {
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
    for (std::size_t position = 0; position < record.rank; ++position) {
      error = ReadSize(reader, &cursor,
                       &record.input_axes_fastest_first[position]);
      if (!error.success()) {
        return error;
      }
    }
    error = ReadSize(reader, &cursor, &record.map_rank);
    if (!error.success()) {
      return error;
    }
    if (record.map_rank > record.rank) {
      return Invalid("reduction map rank exceeds logical rank");
    }
    for (std::size_t position = 0; position < record.map_rank; ++position) {
      error = ReadSize(reader, &cursor, &record.map_loop_order[position]);
      if (!error.success()) {
        return error;
      }
    }
    error = ReadSize(reader, &cursor, &record.reduction_rank);
    if (!error.success()) {
      return error;
    }
    if (record.reduction_rank > record.rank) {
      return Invalid("reduction fiber rank exceeds logical rank");
    }
    for (std::size_t position = 0; position < record.reduction_rank;
         ++position) {
      error = ReadSize(reader, &cursor,
                       &record.reduction_loop_order[position]);
      if (!error.success()) {
        return error;
      }
    }
    error = ValidateReductionAxisOrder(
        record.input_axes_fastest_first, record.rank,
        record.rank == 0 ? 0 : (UINT64_C(1) << record.rank) - 1,
        record.rank, "reduction input axis order");
    if (!error.success()) {
      return error;
    }
    error = ValidateReductionAxisOrder(
        record.map_loop_order, record.map_rank, record.map_axis_mask,
        record.rank, "reduction map loop order");
    if (!error.success()) {
      return error;
    }
    error = ValidateReductionAxisOrder(
        record.reduction_loop_order, record.reduction_rank,
        record.reduction_axis_mask, record.rank,
        "reduction fiber loop order");
    if (!error.success()) {
      return error;
    }
    error = ValidateReductionRecord(&record, plan->input_size,
                                    plan->output_size);
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

enum class PreparedOperation : uint8_t {
  kFreshMap,
  kBaseAssign,
  kBaseAccumulate,
  kSelectedScale,
  kStructuredReduction,
};

uint64_t TrackedExecutionDataBytes(const PreparedExecutionData& data) {
  return sizeof(PreparedExecutionData) +
         data.plan.records.capacity() * sizeof(Record) +
         data.work.capacity() * sizeof(RecordWork);
}

struct PreparedState {
  static ffi::TypeId id;

  PreparedState(std::shared_ptr<const PreparedExecutionData> value,
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

  ~PreparedState() {
    prepared_live_state_count.fetch_sub(1, std::memory_order_relaxed);
    prepared_live_bytes.fetch_sub(tracked_bytes, std::memory_order_relaxed);
    prepared_destroyed_state_count.fetch_add(1,
                                             std::memory_order_relaxed);
  }

  std::shared_ptr<const PreparedExecutionData> execution;
  std::shared_ptr<const ParsedReductionPlan> reduction_execution;
  uint64_t descriptor_bytes = 0;
  uint64_t tracked_bytes = 0;
  PreparedOperation operation = PreparedOperation::kFreshMap;
};

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();

ffi::ErrorOr<std::unique_ptr<PreparedState>> PublishPrepared(
    std::shared_ptr<PreparedExecutionData> execution,
    uint64_t descriptor_size, PreparedOperation operation) {
  const uint64_t tracked_bytes =
      sizeof(PreparedState) + TrackedExecutionDataBytes(*execution);
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
  if (tracked_bytes > std::min(relative_limit, kPreparedAbsoluteBytes)) {
    return ffi::Unexpected(Invalid("prepared state exceeds size limit"));
  }
  return std::make_unique<PreparedState>(
      std::move(execution), descriptor_size, tracked_bytes, operation);
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

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(
    ffi::Span<const uint8_t> descriptor, PreparedOperation operation) {
  prepared_instantiate_count.fetch_add(1, std::memory_order_relaxed);
  auto execution = std::make_shared<PreparedExecutionData>();
  ffi::Error error = ValidateCompiledDescriptor(
      descriptor, &execution->plan, &execution->work,
      &execution->chunks_per_batch,
      &execution->parallel_minimum_bytes);
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
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
      if (!IsUnitScale(record, plan.dtype)) {
        return ffi::Unexpected(
            Invalid("selected scale requires static unit scale"));
      }
    }
  }
  return PublishPrepared(std::move(execution), descriptor.size(), operation);
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePreparedReduction(
    ffi::Span<const uint8_t> descriptor) {
  prepared_instantiate_count.fetch_add(1, std::memory_order_relaxed);
  auto execution = std::make_shared<ParsedReductionPlan>();
  ffi::Error error = ParseReductionDescriptor(descriptor, execution.get());
  if (!error.success()) {
    return ffi::Unexpected(std::move(error));
  }
  return PublishPreparedReduction(std::move(execution), descriptor.size());
}

template <typename SourceElement, typename ResultElement,
          typename RangeFunction>
struct PreparedParallelState {
  PreparedParallelState(
      std::shared_ptr<const PreparedExecutionData> execution_data,
      const SourceElement* source_data, ResultElement* result_data,
      uint64_t batches, uint64_t total_chunks_value,
      RangeFunction range_function)
      : execution(std::move(execution_data)),
        source(source_data),
        result(result_data),
        batch_count(batches),
        total_chunks(total_chunks_value),
        execute_range(std::move(range_function)) {}

  std::shared_ptr<const PreparedExecutionData> execution;
  const SourceElement* source = nullptr;
  ResultElement* result = nullptr;
  uint64_t batch_count = 0;
  uint64_t total_chunks = 0;
  std::atomic<uint64_t> next_chunk{0};
  RangeFunction execute_range;
};

template <typename State>
void ExecutePreparedParallelWorker(const std::shared_ptr<State>& state) {
  const ParsedPlan& plan = state->execution->plan;
  const auto& work = state->execution->work;
  const uint64_t chunks_per_batch =
      state->execution->chunks_per_batch;
  for (;;) {
    const uint64_t global_chunk =
        state->next_chunk.fetch_add(1, std::memory_order_relaxed);
    if (global_chunk >= state->total_chunks) {
      return;
    }
    const uint64_t batch = global_chunk / chunks_per_batch;
    const uint64_t local_chunk =
        global_chunk - batch * chunks_per_batch;
    const auto item = std::upper_bound(
        work.begin(), work.end(), local_chunk,
        [](uint64_t chunk, const RecordWork& candidate) {
          return chunk < candidate.chunk_begin + candidate.chunk_count;
        });
    if (item == work.end()) {
      return;
    }
    const uint64_t chunk_in_record = local_chunk - item->chunk_begin;
    const uint64_t unit_start = chunk_in_record * item->units_per_chunk;
    const uint64_t unit_count = std::min(
        item->units_per_chunk, item->unit_count - unit_start);
    const auto* source_batch =
        plan.source_size == 0
            ? state->source
            : state->source + batch * plan.source_size;
    auto* result_batch =
        plan.output_size == 0
            ? state->result
            : state->result + batch * plan.output_size;
    state->execute_range(plan.records[item->record_index], source_batch,
                         result_batch, item->kind, unit_start,
                         unit_count);
  }
}

enum class OutputInitPolicy : uint8_t {
  kPlanCoverage,
  kAlreadyInitialized,
};

bool KeepDynamicCompactC64Serial(const ParsedPlan& plan,
                                 PreparedOperation operation) {
  if (plan.dtype != kDtypeC64 ||
      operation != PreparedOperation::kSelectedScale) {
    return false;
  }
  for (const Record& record : plan.records) {
    if (!IsCompactSameMapping(record)) {
      return false;
    }
  }
  return true;
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename ExecuteFunction,
          typename RangeFunction>
ffi::Future ExecutePreparedTyped(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result,
    ffi::ThreadPool thread_pool, uint64_t expected_dtype,
    PreparedOperation expected_operation, OutputInitPolicy init_policy,
    ExecuteFunction execute_record, RangeFunction execute_range,
    uint64_t expected_scalar_kind = 0) {
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
                               plan.output_size, &batch_count);
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
  if (init_policy == OutputInitPolicy::kPlanCoverage &&
      plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0) {
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
  if (KeepDynamicCompactC64Serial(plan, expected_operation)) {
    available_workers = std::min<uint64_t>(available_workers, 1);
  }
  const uint64_t threshold_elements =
      init_policy == OutputInitPolicy::kPlanCoverage &&
              plan.coverage == kCoveragePartialUniqueZeroFill
          ? std::max(plan.copied_elements, plan.output_size)
          : plan.copied_elements;
  uint64_t total_threshold_elements = 0;
  uint64_t total_work_bytes = 0;
  if (!CheckedMultiply(threshold_elements, batch_count,
                       &total_threshold_elements) ||
      !CheckedMultiply(total_threshold_elements,
                       std::max(source_item_size, result_item_size),
                       &total_work_bytes)) {
    return ErrorFuture(Invalid("parallel work byte count overflow"));
  }

  const bool has_copy_work =
      batch_count != 0 && plan.copied_elements != 0;
  const bool has_zero_fill_work =
      init_policy == OutputInitPolicy::kPlanCoverage &&
      plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0;
  const bool full_compact_interval =
      plan.records.size() == 1 &&
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
    if (batch_count > 1 && full_compact_interval) {
      Record fused = plan.records[0];
      fused.logical_elements = source_elements;
      execute_record(fused, source_data, result_data);
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
      for (const Record& record : plan.records) {
        execute_record(record, source_batch, result_batch);
      }
    }
    return ReadyFuture();
  };

  if (!has_copy_work || available_workers <= 1 ||
      total_work_bytes < execution->parallel_minimum_bytes) {
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
                                      RangeFunction>;
  auto state = std::make_shared<State>(
      execution, source_data, result_data, batch_count, total_chunks,
      std::move(execute_range));
  last_worker_count.store(worker_count, std::memory_order_relaxed);
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule([state, completion]() mutable {
        try {
          ExecutePreparedParallelWorker(state);
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

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename ScalarOp, typename WriteOp>
ffi::Future ExecutePreparedBaseUpdate(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ResultDtype> base, ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    uint64_t expected_dtype, PreparedOperation operation,
    ScalarOp scalar_op, WriteOp write_op) {
  if (prepared == nullptr || prepared->execution == nullptr) {
    return ErrorFuture(Invalid("prepared state is unavailable"));
  }
  if (descriptor.size() != prepared->descriptor_bytes) {
    return ErrorFuture(Invalid("prepared descriptor byte length mismatch"));
  }
  if (prepared->operation != operation) {
    return ErrorFuture(Invalid("prepared operation does not match FFI target"));
  }
  const ParsedPlan& plan = prepared->execution->plan;
  if (plan.dtype != expected_dtype) {
    return ErrorFuture(
        Invalid("descriptor dtype does not match prepared typed target"));
  }
  uint64_t batch_count = 0;
  ffi::Error error =
      ValidateBufferDimensions(source, result, plan.source_size,
                               plan.output_size, &batch_count);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  const auto base_dims = base.dimensions();
  if (base_dims.size() != 1 || base_dims[0] < 0) {
    return ErrorFuture(Invalid("base must be a rank-one physical buffer"));
  }
  uint64_t result_elements = 0;
  if (!CheckedMultiply(batch_count, plan.output_size, &result_elements) ||
      static_cast<uint64_t>(base_dims[0]) != result_elements) {
    return ErrorFuture(
        Invalid("actual flat base size does not match descriptor"));
  }
  const uint64_t item_size = sizeof(*result->typed_data());
  error = ValidateHostElements(result_elements, item_size, "batched base");
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  error = ValidateDisjointBuffers(base.typed_data(), result->typed_data(),
                                  result_elements, result_elements,
                                  item_size);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  const bool preserve_base =
      operation == PreparedOperation::kBaseAccumulate ||
      plan.coverage == kCoveragePartialUniqueZeroFill;
  if (preserve_base && result_elements != 0) {
    std::memcpy(result->typed_data(), base.typed_data(),
                static_cast<std::size_t>(result_elements) * item_size);
  }
  return ExecutePreparedTyped<SourceDtype, ResultDtype>(
      descriptor, prepared, source, result, thread_pool, expected_dtype,
      operation, OutputInitPolicy::kAlreadyInitialized,
      [scalar_op, write_op](const Record& record, const auto* input,
                            auto* output) {
        ExecuteAffineUpdateRecord(record, input, output, scalar_op,
                                  write_op);
      },
      [scalar_op, write_op](const Record& record, const auto* input,
                            auto* output, RecordWorkKind kind,
                            uint64_t unit_start, uint64_t unit_count) {
        ExecuteAffineUpdateRecordRange(record, input, output, kind,
                                       unit_start, unit_count, scalar_op,
                                       write_op);
      });
}

template <ffi::DataType Dtype, typename DynamicScalarOp>
ffi::Future ExecutePreparedSelectedScale(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<Dtype> base, ffi::Buffer<Dtype> factor,
    ffi::ResultBuffer<Dtype> result, ffi::ThreadPool thread_pool,
    uint64_t expected_dtype, DynamicScalarOp dynamic_scalar_op) {
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
                               plan.output_size, &batch_count);
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
  error = ValidateDisjointBuffers(base.typed_data(), result->typed_data(),
                                  result_elements, result_elements,
                                  item_size);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  error = ValidateDisjointBuffers(factor.typed_data(), result->typed_data(),
                                  factor_count, result_elements, item_size);
  if (!error.success()) {
    return ErrorFuture(std::move(error));
  }
  if (plan.coverage == kCoveragePartialUniqueZeroFill &&
      result_elements != 0) {
    std::memcpy(result->typed_data(), base.typed_data(),
                static_cast<std::size_t>(result_elements) * item_size);
  }

  const auto* factor_data = factor.typed_data();
  auto* result_data = result->typed_data();
  const auto assign = [](auto& destination, auto mapped) {
    destination = mapped;
  };
  const uint64_t output_size = plan.output_size;
  const auto factor_for_output =
      [factor_data, result_data, factor_count,
       output_size](const auto* output) {
        uint64_t batch = 0;
        if (factor_count != 1 && output_size != 0) {
          batch = static_cast<uint64_t>(output - result_data) / output_size;
        }
        return factor_data[factor_count == 1 ? 0 : batch];
      };
  const auto execute_compact =
      [dynamic_scalar_op](const auto* input, auto* output,
                          uint64_t element_count, auto factor_value) {
        if constexpr (Dtype == ffi::C64) {
          ExecuteDynamicScaledC64Range(input, output, element_count,
                                       factor_value);
        } else {
          for (uint64_t element = 0; element < element_count; ++element) {
            output[element] =
                dynamic_scalar_op(factor_value, input[element]);
          }
        }
      };
  const auto execute_record =
      [factor_count, batch_count, output_size, result_data,
       factor_data, factor_for_output, execute_compact, dynamic_scalar_op,
       assign](const Record& record, const auto* input, auto* output) {
        if (factor_count != 1 && output == result_data &&
            record.logical_elements == batch_count * output_size) {
          for (uint64_t batch = 0; batch < batch_count; ++batch) {
            execute_compact(input + batch * output_size,
                            output + batch * output_size, output_size,
                            factor_data[batch]);
          }
          return;
        }
        const auto factor_value = factor_for_output(output);
        if (IsCompactSameMapping(record)) {
          execute_compact(input + record.source_offset,
                          output + record.destination_offset,
                          record.logical_elements, factor_value);
          return;
        }
        const auto scalar_op = [dynamic_scalar_op, factor_value](
                                   auto value, const Record&) {
          return dynamic_scalar_op(factor_value, value);
        };
        ExecuteAffineUpdateRecord(record, input, output, scalar_op, assign);
      };
  const auto execute_range =
      [factor_for_output, execute_compact, dynamic_scalar_op,
       assign](const Record& record, const auto* input, auto* output,
               RecordWorkKind kind, uint64_t unit_start,
               uint64_t unit_count) {
        const auto factor_value = factor_for_output(output);
        if (kind == RecordWorkKind::kCompact) {
          execute_compact(input + record.source_offset + unit_start,
                          output + record.destination_offset + unit_start,
                          unit_count, factor_value);
          return;
        }
        const auto scalar_op = [dynamic_scalar_op, factor_value](
                                   auto value, const Record&) {
          return dynamic_scalar_op(factor_value, value);
        };
        ExecuteAffineUpdateRecordRange(record, input, output, kind,
                                       unit_start, unit_count, scalar_op,
                                       assign);
      };
  return ExecutePreparedTyped<Dtype, Dtype>(
      descriptor, prepared, base, result, thread_pool, expected_dtype,
      PreparedOperation::kSelectedScale,
      OutputInitPolicy::kAlreadyInitialized, execute_record,
      execute_range);
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
        "tensor0-stride: contained ABI-v7 instantiate exception"));
  } catch (...) {
    return ffi::Unexpected(ffi::Error::Internal(
        "tensor0-stride: contained unknown ABI-v7 instantiate exception"));
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
    return ExecutePreparedTyped<ffi::F32, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const float* input, float* output) {
          ExecuteRecord(record, input, output);
        },
        [](const Record& record, const float* input, float* output,
           RecordWorkKind kind, uint64_t unit_start,
           uint64_t unit_count) {
          ExecuteRecordRange(record, input, output, kind, unit_start,
                             unit_count);
        });
  });
}

ffi::Future ExecutePreparedF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F16> source, ffi::ResultBuffer<ffi::F16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedTyped<ffi::F16, ffi::F16>(
        descriptor, prepared, source, result, thread_pool, kDtypeF16,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const uint16_t* input,
           uint16_t* output) {
          ExecuteScalarRecord(record, input, output,
                              record.scale_real_bits == 0x3C00U,
                              ScaleF16);
        },
        [](const Record& record, const uint16_t* input,
           uint16_t* output, RecordWorkKind kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteScalarRecordRange(
              record, input, output,
              record.scale_real_bits == 0x3C00U, ScaleF16,
              kind == RecordWorkKind::kCompact, unit_start,
              unit_count);
        });
  });
}

ffi::Future ExecutePreparedBF16Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::BF16> source, ffi::ResultBuffer<ffi::BF16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedTyped<ffi::BF16, ffi::BF16>(
        descriptor, prepared, source, result, thread_pool, kDtypeBF16,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const uint16_t* input,
           uint16_t* output) {
          ExecuteScalarRecord(record, input, output,
                              record.scale_real_bits == 0x3F80U,
                              ScaleBF16);
        },
        [](const Record& record, const uint16_t* input,
           uint16_t* output, RecordWorkKind kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteScalarRecordRange(
              record, input, output,
              record.scale_real_bits == 0x3F80U, ScaleBF16,
              kind == RecordWorkKind::kCompact, unit_start,
              unit_count);
        });
  });
}

ffi::Future ExecutePreparedC64Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedTyped<ffi::C64, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const std::complex<float>* input,
           std::complex<float>* output) {
          ExecuteC64Record(record, input, output);
        },
        [](const Record& record, const std::complex<float>* input,
           std::complex<float>* output, RecordWorkKind kind,
           uint64_t unit_start, uint64_t unit_count) {
          ExecuteC64RecordRange(record, input, output, kind, unit_start,
                                unit_count);
        });
  });
}

ffi::Future ExecutePreparedS32Impl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::S32> source, ffi::ResultBuffer<ffi::S32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedTyped<ffi::S32, ffi::S32>(
        descriptor, prepared, source, result, thread_pool, kDtypeS32,
        PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
        [](const Record& record, const int32_t* input,
           int32_t* output) {
          ExecuteScalarRecord(record, input, output,
                              record.scale_real_bits == 1U, ScaleS32);
        },
        [](const Record& record, const int32_t* input, int32_t* output,
           RecordWorkKind kind, uint64_t unit_start,
           uint64_t unit_count) {
          ExecuteScalarRecordRange(
              record, input, output, record.scale_real_bits == 1U,
              ScaleS32, kind == RecordWorkKind::kCompact, unit_start,
              unit_count);
        });
  });
}

#define TENSOR0_PREPARED_SCALAR_IMPL(Name, Dtype, DtypeCode, ScaleFunction)  \
  ffi::Future Name(                                                          \
      ffi::Span<const uint8_t> descriptor, PreparedState* prepared,          \
      ffi::Buffer<Dtype> source, ffi::ResultBuffer<Dtype> result,            \
      ffi::ThreadPool thread_pool) {                                         \
    return ContainExceptions([&] {                                           \
      return ExecutePreparedTyped<Dtype, Dtype>(                             \
          descriptor, prepared, source, result, thread_pool, DtypeCode,      \
          PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,     \
          [](const Record& record, const auto* input, auto* output) {         \
            ExecuteScalarRecord(record, input, output,                       \
                                IsUnitScale(record, DtypeCode),              \
                                ScaleFunction);                              \
          },                                                                 \
          [](const Record& record, const auto* input, auto* output,           \
             RecordWorkKind kind, uint64_t unit_start,                       \
             uint64_t unit_count) {                                          \
            ExecuteScalarRecordRange(                                       \
                record, input, output, IsUnitScale(record, DtypeCode),       \
                ScaleFunction, kind == RecordWorkKind::kCompact,             \
                unit_start, unit_count);                                     \
          });                                                                \
    });                                                                      \
  }

TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedPredImpl, ffi::PRED,
                             kDtypePred, ScalePred)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS8Impl, ffi::S8, kDtypeS8,
                             ScaleInteger<int8_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS16Impl, ffi::S16, kDtypeS16,
                             ScaleInteger<int16_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedS64Impl, ffi::S64, kDtypeS64,
                             ScaleInteger<int64_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU8Impl, ffi::U8, kDtypeU8,
                             ScaleInteger<uint8_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU16Impl, ffi::U16, kDtypeU16,
                             ScaleInteger<uint16_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU32Impl, ffi::U32, kDtypeU32,
                             ScaleInteger<uint32_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedU64Impl, ffi::U64, kDtypeU64,
                             ScaleInteger<uint64_t>)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedF64Impl, ffi::F64, kDtypeF64,
                             ScaleF64)
TENSOR0_PREPARED_SCALAR_IMPL(ExecutePreparedC128Impl, ffi::C128, kDtypeC128,
                             ScaleC128)

#undef TENSOR0_PREPARED_SCALAR_IMPL

float ScaleBitsF32(const Record& record) {
  return BitCast<float>(static_cast<uint32_t>(record.scale_real_bits));
}

double ScaleBitsF64(const Record& record) {
  return BitCast<double>(record.scale_real_bits);
}

std::complex<double> ScaleBitsC128(const Record& record) {
  return {BitCast<double>(record.scale_real_bits),
          BitCast<double>(record.scale_imaginary_bits)};
}

std::complex<float> ScaleBitsC64(const Record& record) {
  return {BitCast<float>(static_cast<uint32_t>(record.scale_real_bits)),
          BitCast<float>(static_cast<uint32_t>(record.scale_imaginary_bits))};
}

template <ffi::DataType SourceDtype, ffi::DataType ResultDtype,
          typename ScalarOp>
ffi::Future ExecutePreparedMixed(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    uint64_t scalar_dtype, uint64_t scalar_kind, ScalarOp scalar_op) {
  return ExecutePreparedTyped<SourceDtype, ResultDtype>(
      descriptor, prepared, source, result, thread_pool, scalar_dtype,
      PreparedOperation::kFreshMap, OutputInitPolicy::kPlanCoverage,
      [scalar_op](const Record& record, const auto* input, auto* output) {
        ExecuteAffineRecord(record, input, output, scalar_op);
      },
      [scalar_op](const Record& record, const auto* input, auto* output,
                  RecordWorkKind kind, uint64_t unit_start,
                  uint64_t unit_count) {
        ExecuteAffineRecordRange(record, input, output,
                                 kind == RecordWorkKind::kCompact,
                                 unit_start, unit_count, scalar_op);
      },
      scalar_kind);
}

ffi::Future ExecutePreparedF16F32ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F16> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F16, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarForwardScaleCast,
        [](uint16_t value, const Record& record) {
          return HalfToFloat(value) * ScaleBitsF32(record);
        });
  });
}

ffi::Future ExecutePreparedF32F16TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::F16> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::F16>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarJaxTranspose,
        [](float value, const Record& record) {
          return FloatToHalf(value * ScaleBitsF32(record));
        });
  });
}

ffi::Future ExecutePreparedF32C64ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        kScalarForwardScaleCast,
        [](float value, const Record& record) {
          const std::complex<float> scale = ScaleBitsC64(record);
          return MultiplyC64(scale.real(), scale.imag(), value, 0.0F);
        });
  });
}

ffi::Future ExecutePreparedC64F32TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C64, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeC64,
        kScalarJaxTranspose,
        [](std::complex<float> value, const Record& record) {
          const std::complex<float> scale = ScaleBitsC64(record);
          return RealPartMultiplyC64(scale.real(), scale.imag(), value.real(),
                                     value.imag());
        });
  });
}

ffi::Future ExecutePreparedC64F32ForwardImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::C64> source, ffi::ResultBuffer<ffi::F32> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::C64, ffi::F32>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarForwardScaleCast,
        [](std::complex<float> value, const Record& record) {
          return value.real() * ScaleBitsF32(record);
        });
  });
}

ffi::Future ExecutePreparedF32C64TransposeImpl(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ffi::F32> source, ffi::ResultBuffer<ffi::C64> result,
    ffi::ThreadPool thread_pool) {
  return ContainExceptions([&] {
    return ExecutePreparedMixed<ffi::F32, ffi::C64>(
        descriptor, prepared, source, result, thread_pool, kDtypeF32,
        kScalarJaxTranspose,
        [](float value, const Record& record) {
          return std::complex<float>(value * ScaleBitsF32(record), 0.0F);
        });
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
        [](double value, const Record& record) {
          const std::complex<double> scale = ScaleBitsC128(record);
          return MultiplyC128(scale.real(), scale.imag(), value, 0.0);
        });
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
        [](std::complex<double> value, const Record& record) {
          const std::complex<double> scale = ScaleBitsC128(record);
          return RealPartMultiplyC128(scale.real(), scale.imag(), value.real(),
                                      value.imag());
        });
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
        [](std::complex<double> value, const Record& record) {
          return value.real() * ScaleBitsF64(record);
        });
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
        [](double value, const Record& record) {
          return std::complex<double>(value * ScaleBitsF64(record), 0.0);
        });
  });
}

struct ScaleSameF32 {
  float operator()(float value, const Record& record) const {
    if (record.scale_real_bits == 0x3F800000U) {
      return value;
    }
    return ScaleBitsF32(record) * value;
  }
};

struct ScaleSameF16 {
  uint16_t operator()(uint16_t value, const Record& record) const {
    if (record.scale_real_bits == 0x3C00U) {
      return value;
    }
    return ScaleF16(value, record);
  }
};

struct ScaleSameBF16 {
  uint16_t operator()(uint16_t value, const Record& record) const {
    if (record.scale_real_bits == 0x3F80U) {
      return value;
    }
    return ScaleBF16(value, record);
  }
};

struct ScaleSameC64 {
  std::complex<float> operator()(std::complex<float> value,
                                 const Record& record) const {
    if (record.scale_real_bits == 0x3F800000U &&
        record.scale_imaginary_bits == 0) {
      return value;
    }
    return ScaleC64(value, record);
  }
};

struct ScaleSamePred {
  bool operator()(bool value, const Record& record) const {
    return IsUnitScale(record, kDtypePred) ? value : ScalePred(value, record);
  }
};

template <typename Element>
struct ScaleSameInteger {
  Element operator()(Element value, const Record& record) const {
    return record.scale_real_bits == 1U ? value
                                        : ScaleInteger(value, record);
  }
};

struct ScaleSameF64 {
  double operator()(double value, const Record& record) const {
    return IsUnitScale(record, kDtypeF64) ? value : ScaleF64(value, record);
  }
};

struct ScaleSameC128 {
  std::complex<double> operator()(std::complex<double> value,
                                  const Record& record) const {
    return IsUnitScale(record, kDtypeC128) ? value
                                           : ScaleC128(value, record);
  }
};

struct ScaleF16F32 {
  float operator()(uint16_t value, const Record& record) const {
    return HalfToFloat(value) * ScaleBitsF32(record);
  }
};

struct ScaleF32C64 {
  std::complex<float> operator()(float value, const Record& record) const {
    const std::complex<float> scale = ScaleBitsC64(record);
    return MultiplyC64(scale.real(), scale.imag(), value, 0.0F);
  }
};

struct ScaleC64F32 {
  float operator()(std::complex<float> value, const Record& record) const {
    return value.real() * ScaleBitsF32(record);
  }
};

struct ScaleF64C128 {
  std::complex<double> operator()(double value, const Record& record) const {
    const std::complex<double> scale = ScaleBitsC128(record);
    return MultiplyC128(scale.real(), scale.imag(), value, 0.0);
  }
};

struct ScaleC128F64 {
  double operator()(std::complex<double> value,
                    const Record& record) const {
    return value.real() * ScaleBitsF64(record);
  }
};

struct ScaleS32Pred {
  bool operator()(int32_t value, const Record& record) const {
    return record.scale_real_bits != 0 && value != 0;
  }
};

template <typename Result>
struct ScaleS32Narrow {
  Result operator()(int32_t value, const Record& record) const {
    using ResultUnsigned = std::make_unsigned_t<Result>;
    const Result scale = BitCast<Result>(
        static_cast<ResultUnsigned>(record.scale_real_bits));
    const uint32_t product =
        BitCast<uint32_t>(value) * BitCast<uint32_t>(static_cast<int32_t>(scale));
    return BitCast<Result>(static_cast<ResultUnsigned>(product));
  }
};

struct ScaleU32U8 {
  uint8_t operator()(uint32_t value, const Record& record) const {
    return static_cast<uint8_t>(
        value * static_cast<uint32_t>(record.scale_real_bits));
  }
};

struct AddF32 {
  float operator()(float left, float right) const { return left + right; }
};

struct AddF16 {
  uint16_t operator()(uint16_t left, uint16_t right) const {
    return FloatToHalf(HalfToFloat(left) + HalfToFloat(right));
  }
};

struct AddBF16 {
  uint16_t operator()(uint16_t left, uint16_t right) const {
    return FloatToBFloat16(BFloat16ToFloat(left) + BFloat16ToFloat(right));
  }
};

struct AddC64 {
  std::complex<float> operator()(std::complex<float> left,
                                 std::complex<float> right) const {
    return {left.real() + right.real(), left.imag() + right.imag()};
  }
};

struct AddPred {
  bool operator()(bool left, bool right) const { return left || right; }
};

template <typename Element>
struct AddIntegerOp {
  Element operator()(Element left, Element right) const {
    return AddInteger(left, right);
  }
};

struct AddF64 {
  double operator()(double left, double right) const { return left + right; }
};

struct AddC128 {
  std::complex<double> operator()(std::complex<double> left,
                                  std::complex<double> right) const {
    return {left.real() + right.real(), left.imag() + right.imag()};
  }
};

template <PreparedOperation kOperation, ffi::DataType SourceDtype,
          ffi::DataType ResultDtype, typename ScalarOp, typename AddOp>
ffi::Future ExecuteBaseUpdateOperation(
    ffi::Span<const uint8_t> descriptor, PreparedState* prepared,
    ffi::Buffer<ResultDtype> base, ffi::Buffer<SourceDtype> source,
    ffi::ResultBuffer<ResultDtype> result, ffi::ThreadPool thread_pool,
    uint64_t expected_dtype, ScalarOp scalar_op, AddOp add_op) {
  const auto write_op = [add_op](auto& destination, auto mapped) {
    if constexpr (kOperation == PreparedOperation::kBaseAssign) {
      destination = mapped;
    } else {
      destination = add_op(destination, mapped);
    }
  };
  return ExecutePreparedBaseUpdate<SourceDtype, ResultDtype>(
      descriptor, prepared, base, source, result, thread_pool,
      expected_dtype, kOperation, scalar_op, write_op);
}

#define TENSOR0_BASE_UPDATE_IMPL(Name, Operation, SourceType, ResultType,     \
                                 DtypeCode, ScalarType, AddType)              \
  ffi::Future Name(                                                           \
      ffi::Span<const uint8_t> descriptor, PreparedState* prepared,           \
      ffi::Buffer<ResultType> base, ffi::Buffer<SourceType> source,           \
      ffi::ResultBuffer<ResultType> result, ffi::ThreadPool thread_pool) {    \
    return ContainExceptions([&] {                                            \
      return ExecuteBaseUpdateOperation<Operation, SourceType, ResultType>(   \
          descriptor, prepared, base, source, result, thread_pool, DtypeCode, \
          ScalarType{}, AddType{});                                           \
    });                                                                       \
  }

#define TENSOR0_BASE_UPDATE_DTYPES(Prefix, Operation)                         \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F32Impl, Operation, ffi::F32, ffi::F32,    \
                           kDtypeF32, ScaleSameF32, AddF32)                    \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F16Impl, Operation, ffi::F16, ffi::F16,    \
                           kDtypeF16, ScaleSameF16, AddF16)                    \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##BF16Impl, Operation, ffi::BF16, ffi::BF16, \
                           kDtypeBF16, ScaleSameBF16, AddBF16)                 \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##C64Impl, Operation, ffi::C64, ffi::C64,    \
                           kDtypeC64, ScaleSameC64, AddC64)                    \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S32Impl, Operation, ffi::S32, ffi::S32,    \
                           kDtypeS32, ScaleSameInteger<int32_t>,               \
                           AddIntegerOp<int32_t>)                              \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##PredImpl, Operation, ffi::PRED, ffi::PRED, \
                           kDtypePred, ScaleSamePred, AddPred)                 \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S8Impl, Operation, ffi::S8, ffi::S8,       \
                           kDtypeS8, ScaleSameInteger<int8_t>,                 \
                           AddIntegerOp<int8_t>)                               \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S16Impl, Operation, ffi::S16, ffi::S16,    \
                           kDtypeS16, ScaleSameInteger<int16_t>,               \
                           AddIntegerOp<int16_t>)                              \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S64Impl, Operation, ffi::S64, ffi::S64,    \
                           kDtypeS64, ScaleSameInteger<int64_t>,               \
                           AddIntegerOp<int64_t>)                              \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##U8Impl, Operation, ffi::U8, ffi::U8,       \
                           kDtypeU8, ScaleSameInteger<uint8_t>,                \
                           AddIntegerOp<uint8_t>)                              \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##U16Impl, Operation, ffi::U16, ffi::U16,    \
                           kDtypeU16, ScaleSameInteger<uint16_t>,              \
                           AddIntegerOp<uint16_t>)                             \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##U32Impl, Operation, ffi::U32, ffi::U32,    \
                           kDtypeU32, ScaleSameInteger<uint32_t>,              \
                           AddIntegerOp<uint32_t>)                             \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##U64Impl, Operation, ffi::U64, ffi::U64,    \
                           kDtypeU64, ScaleSameInteger<uint64_t>,              \
                           AddIntegerOp<uint64_t>)                             \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F64Impl, Operation, ffi::F64, ffi::F64,    \
                           kDtypeF64, ScaleSameF64, AddF64)                    \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##C128Impl, Operation, ffi::C128, ffi::C128, \
                           kDtypeC128, ScaleSameC128, AddC128)                 \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F16F32Impl, Operation, ffi::F16, ffi::F32, \
                           kDtypeF32, ScaleF16F32, AddF32)                     \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F32C64Impl, Operation, ffi::F32, ffi::C64, \
                           kDtypeC64, ScaleF32C64, AddC64)                     \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##C64F32Impl, Operation, ffi::C64, ffi::F32, \
                           kDtypeF32, ScaleC64F32, AddF32)                     \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##F64C128Impl, Operation, ffi::F64,          \
                           ffi::C128, kDtypeC128, ScaleF64C128, AddC128)       \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##C128F64Impl, Operation, ffi::C128,         \
                           ffi::F64, kDtypeF64, ScaleC128F64, AddF64)          \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S32PredImpl, Operation, ffi::S32,          \
                           ffi::PRED, kDtypePred, ScaleS32Pred, AddPred)       \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S32S8Impl, Operation, ffi::S32, ffi::S8,   \
                           kDtypeS8, ScaleS32Narrow<int8_t>,                   \
                           AddIntegerOp<int8_t>)                              \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##S32S16Impl, Operation, ffi::S32, ffi::S16, \
                           kDtypeS16, ScaleS32Narrow<int16_t>,                 \
                           AddIntegerOp<int16_t>)                             \
  TENSOR0_BASE_UPDATE_IMPL(Prefix##U32U8Impl, Operation, ffi::U32, ffi::U8,   \
                           kDtypeU8, ScaleU32U8, AddIntegerOp<uint8_t>)

TENSOR0_BASE_UPDATE_DTYPES(ExecuteBaseAssign, PreparedOperation::kBaseAssign)
TENSOR0_BASE_UPDATE_DTYPES(ExecuteBaseAccumulate,
                           PreparedOperation::kBaseAccumulate)

#undef TENSOR0_BASE_UPDATE_DTYPES
#undef TENSOR0_BASE_UPDATE_IMPL

struct DynamicScaleF32 {
  float operator()(float factor, float value) const { return factor * value; }
};

struct DynamicScaleF16 {
  uint16_t operator()(uint16_t factor, uint16_t value) const {
    return FloatToHalf(HalfToFloat(factor) * HalfToFloat(value));
  }
};

struct DynamicScaleBF16 {
  uint16_t operator()(uint16_t factor, uint16_t value) const {
    return FloatToBFloat16(BFloat16ToFloat(factor) *
                           BFloat16ToFloat(value));
  }
};

struct DynamicScaleC64 {
  std::complex<float> operator()(std::complex<float> factor,
                                 std::complex<float> value) const {
    return MultiplyC64(factor.real(), factor.imag(), value.real(),
                       value.imag());
  }
};

struct DynamicScalePred {
  bool operator()(bool factor, bool value) const { return factor && value; }
};

template <typename Element>
struct DynamicScaleInteger {
  Element operator()(Element factor, Element value) const {
    using Unsigned = std::make_unsigned_t<Element>;
    const Unsigned product = static_cast<Unsigned>(
        static_cast<uint64_t>(BitCast<Unsigned>(factor)) *
        static_cast<uint64_t>(BitCast<Unsigned>(value)));
    return BitCast<Element>(product);
  }
};

struct DynamicScaleF64 {
  double operator()(double factor, double value) const {
    return factor * value;
  }
};

struct DynamicScaleC128 {
  std::complex<double> operator()(std::complex<double> factor,
                                  std::complex<double> value) const {
    return MultiplyC128(factor.real(), factor.imag(), value.real(),
                        value.imag());
  }
};

#define TENSOR0_SELECTED_SCALE_IMPL(Name, Dtype, DtypeCode, ScalarType)       \
  ffi::Future Name(                                                           \
      ffi::Span<const uint8_t> descriptor, PreparedState* prepared,           \
      ffi::Buffer<Dtype> base, ffi::Buffer<Dtype> factor,                     \
      ffi::ResultBuffer<Dtype> result, ffi::ThreadPool thread_pool) {         \
    return ContainExceptions([&] {                                            \
      return ExecutePreparedSelectedScale<Dtype>(                             \
          descriptor, prepared, base, factor, result, thread_pool, DtypeCode, \
          ScalarType{});                                                       \
    });                                                                       \
  }

TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleF32Impl, ffi::F32,
                            kDtypeF32, DynamicScaleF32)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleF16Impl, ffi::F16,
                            kDtypeF16, DynamicScaleF16)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleBF16Impl, ffi::BF16,
                            kDtypeBF16, DynamicScaleBF16)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleC64Impl, ffi::C64,
                            kDtypeC64, DynamicScaleC64)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleS32Impl, ffi::S32,
                            kDtypeS32, DynamicScaleInteger<int32_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScalePredImpl, ffi::PRED,
                            kDtypePred, DynamicScalePred)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleS8Impl, ffi::S8,
                            kDtypeS8, DynamicScaleInteger<int8_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleS16Impl, ffi::S16,
                            kDtypeS16, DynamicScaleInteger<int16_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleS64Impl, ffi::S64,
                            kDtypeS64, DynamicScaleInteger<int64_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleU8Impl, ffi::U8,
                            kDtypeU8, DynamicScaleInteger<uint8_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleU16Impl, ffi::U16,
                            kDtypeU16, DynamicScaleInteger<uint16_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleU32Impl, ffi::U32,
                            kDtypeU32, DynamicScaleInteger<uint32_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleU64Impl, ffi::U64,
                            kDtypeU64, DynamicScaleInteger<uint64_t>)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleF64Impl, ffi::F64,
                            kDtypeF64, DynamicScaleF64)
TENSOR0_SELECTED_SCALE_IMPL(ExecuteSelectedScaleC128Impl, ffi::C128,
                            kDtypeC128, DynamicScaleC128)

#undef TENSOR0_SELECTED_SCALE_IMPL

float ReductionScaleBitsF32(const ReductionRecord& record) {
  return BitCast<float>(static_cast<uint32_t>(record.scale_real_bits));
}

std::complex<float> ReductionScaleBitsC64(const ReductionRecord& record) {
  return {BitCast<float>(static_cast<uint32_t>(record.scale_real_bits)),
          BitCast<float>(static_cast<uint32_t>(record.scale_imaginary_bits))};
}

double ReductionScaleBitsF64(const ReductionRecord& record) {
  return BitCast<double>(record.scale_real_bits);
}

std::complex<double> ReductionScaleBitsC128(
    const ReductionRecord& record) {
  return {BitCast<double>(record.scale_real_bits),
          BitCast<double>(record.scale_imaginary_bits)};
}

struct ReductionScaleF32 {
  float operator()(float value, const ReductionRecord& record) const {
    return ReductionScaleBitsF32(record) * value;
  }
};

struct ReductionScaleF16 {
  uint16_t operator()(uint16_t value, const ReductionRecord& record) const {
    return FloatToHalf(HalfToFloat(record.scale_real_bits) *
                       HalfToFloat(value));
  }
};

struct ReductionScaleBF16 {
  uint16_t operator()(uint16_t value, const ReductionRecord& record) const {
    return FloatToBFloat16(BFloat16ToFloat(record.scale_real_bits) *
                           BFloat16ToFloat(value));
  }
};

struct ReductionScaleC64 {
  std::complex<float> operator()(std::complex<float> value,
                                 const ReductionRecord& record) const {
    const std::complex<float> scale = ReductionScaleBitsC64(record);
    return MultiplyC64(scale.real(), scale.imag(), value.real(),
                       value.imag());
  }
};

struct ReductionScaleF32F16Transpose {
  uint16_t operator()(float value, const ReductionRecord& record) const {
    return FloatToHalf(ReductionScaleBitsF32(record) * value);
  }
};

struct ReductionScaleC64F32Transpose {
  float operator()(std::complex<float> value,
                   const ReductionRecord& record) const {
    const std::complex<float> scale = ReductionScaleBitsC64(record);
    return RealPartMultiplyC64(scale.real(), scale.imag(), value.real(),
                               value.imag());
  }
};

struct ReductionScaleF32C64Transpose {
  std::complex<float> operator()(float value,
                                 const ReductionRecord& record) const {
    return {ReductionScaleBitsF32(record) * value, 0.0F};
  }
};

struct ReductionScaleF16F32Forward {
  float operator()(uint16_t value, const ReductionRecord& record) const {
    return HalfToFloat(value) * ReductionScaleBitsF32(record);
  }
};

struct ReductionScaleF32C64Forward {
  std::complex<float> operator()(float value,
                                 const ReductionRecord& record) const {
    const std::complex<float> scale = ReductionScaleBitsC64(record);
    return MultiplyC64(scale.real(), scale.imag(), value, 0.0F);
  }
};

struct ReductionScaleC64F32Forward {
  float operator()(std::complex<float> value,
                   const ReductionRecord& record) const {
    return value.real() * ReductionScaleBitsF32(record);
  }
};

struct ReductionScalePred {
  bool operator()(bool value, const ReductionRecord& record) const {
    return value && record.scale_real_bits != 0;
  }
};

template <typename Element>
struct ReductionScaleInteger {
  Element operator()(Element value, const ReductionRecord& record) const {
    using Unsigned = std::make_unsigned_t<Element>;
    const Unsigned input = BitCast<Unsigned>(value);
    const Unsigned scale = static_cast<Unsigned>(record.scale_real_bits);
    const Unsigned product = static_cast<Unsigned>(
        static_cast<uint64_t>(input) * static_cast<uint64_t>(scale));
    return BitCast<Element>(product);
  }
};

struct ReductionScaleF64 {
  double operator()(double value, const ReductionRecord& record) const {
    return ReductionScaleBitsF64(record) * value;
  }
};

struct ReductionScaleC128 {
  std::complex<double> operator()(std::complex<double> value,
                                  const ReductionRecord& record) const {
    const std::complex<double> scale = ReductionScaleBitsC128(record);
    return MultiplyC128(scale.real(), scale.imag(), value.real(),
                        value.imag());
  }
};

struct ReductionScaleC128F64Transpose {
  double operator()(std::complex<double> value,
                    const ReductionRecord& record) const {
    const std::complex<double> scale = ReductionScaleBitsC128(record);
    return RealPartMultiplyC128(scale.real(), scale.imag(), value.real(),
                                value.imag());
  }
};

struct ReductionScaleF64C128Transpose {
  std::complex<double> operator()(double value,
                                  const ReductionRecord& record) const {
    return {ReductionScaleBitsF64(record) * value, 0.0};
  }
};

struct ReductionScaleF64C128Forward {
  std::complex<double> operator()(double value,
                                  const ReductionRecord& record) const {
    const std::complex<double> scale = ReductionScaleBitsC128(record);
    return MultiplyC128(scale.real(), scale.imag(), value, 0.0);
  }
};

struct ReductionScaleC128F64Forward {
  double operator()(std::complex<double> value,
                    const ReductionRecord& record) const {
    return value.real() * ReductionScaleBitsF64(record);
  }
};

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename AddOp>
void ExecuteReductionOutputRange(const ReductionRecord& record,
                                 const SourceElement* source,
                                 ResultElement* result,
                                 uint64_t output_start,
                                 uint64_t output_count, ScalarOp scalar_op,
                                 AddOp add_op) {
  if (output_count == 0 || record.reduction_count == 0) {
    return;
  }
  const uint64_t output_stop = output_start + output_count;
  for (uint64_t output_linear = output_start;
       output_linear < output_stop; ++output_linear) {
    uint64_t remaining = output_linear;
    int64_t input_base = record.input_offset;
    int64_t output_address = record.output_offset;
    for (std::size_t position = 0; position < record.map_rank; ++position) {
      const std::size_t axis = record.map_loop_order[position];
      const uint64_t coordinate = remaining % record.shape[axis];
      remaining /= record.shape[axis];
      input_base += static_cast<int64_t>(coordinate) *
                    record.input_strides[axis];
      output_address += static_cast<int64_t>(coordinate) *
                        record.output_strides[axis];
    }

    ResultElement accumulator{};
    std::array<uint64_t, kMaximumRank> coordinates{};
    int64_t input_address = input_base;
    for (uint64_t reduction_linear = 0;
         reduction_linear < record.reduction_count; ++reduction_linear) {
      accumulator = add_op(
          accumulator,
          scalar_op(source[input_address], record));
      if (reduction_linear + 1 == record.reduction_count) {
        break;
      }
      for (std::size_t position = 0; position < record.reduction_rank;
           ++position) {
        const std::size_t axis = record.reduction_loop_order[position];
        if (coordinates[axis] + 1 < record.shape[axis]) {
          ++coordinates[axis];
          input_address += record.input_strides[axis];
          break;
        }
        input_address -= static_cast<int64_t>(coordinates[axis]) *
                         record.input_strides[axis];
        coordinates[axis] = 0;
      }
    }
    result[output_address] =
        add_op(result[output_address], accumulator);
  }
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename AddOp>
struct ReductionParallelState {
  std::shared_ptr<const ParsedReductionPlan> execution;
  const SourceElement* source = nullptr;
  ResultElement* result = nullptr;
  uint64_t batch_count = 0;
  uint64_t chunks_per_batch = 0;
  uint64_t total_chunks = 0;
  uint64_t outputs_per_chunk = 0;
  std::atomic<uint64_t> next_chunk{0};
  ScalarOp scalar_op;
  AddOp add_op;
};

template <typename State>
void ExecuteReductionParallelWorker(const std::shared_ptr<State>& state) {
  const ParsedReductionPlan& plan = *state->execution;
  const ReductionRecord& record = plan.records.front();
  for (;;) {
    const uint64_t global_chunk =
        state->next_chunk.fetch_add(1, std::memory_order_relaxed);
    if (global_chunk >= state->total_chunks) {
      return;
    }
    const uint64_t batch = global_chunk / state->chunks_per_batch;
    const uint64_t local_chunk =
        global_chunk - batch * state->chunks_per_batch;
    const uint64_t output_start =
        local_chunk * state->outputs_per_chunk;
    const uint64_t output_count = std::min(
        state->outputs_per_chunk, record.output_count - output_start);
    const auto* source_batch =
        plan.input_size == 0
            ? state->source
            : state->source + batch * plan.input_size;
    auto* result_batch =
        plan.output_size == 0
            ? state->result
            : state->result + batch * plan.output_size;
    ExecuteReductionOutputRange(record, source_batch, result_batch,
                                output_start, output_count,
                                state->scalar_op, state->add_op);
  }
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
  if (result_elements != 0) {
    std::fill(result_data, result_data + result_elements, ResultElement{});
  }
  const int64_t pool_threads = thread_pool.num_threads();
  const uint64_t raw_available_workers =
      pool_threads > 0 ? static_cast<uint64_t>(pool_threads) : 0;
  last_available_worker_count.store(raw_available_workers,
                                    std::memory_order_relaxed);
  const uint64_t available_workers = std::min(
      raw_available_workers,
      worker_limit.load(std::memory_order_relaxed));

  const auto execute_sequential = [&] {
    const bool has_work = batch_count != 0 &&
                          std::any_of(
                              plan.records.begin(), plan.records.end(),
                              [](const ReductionRecord& record) {
                                return record.output_count != 0 &&
                                       record.reduction_count != 0;
                              });
    last_worker_count.store(has_work || result_elements != 0 ? 1 : 0,
                            std::memory_order_relaxed);
    for (uint64_t batch = 0; batch < batch_count; ++batch) {
      const auto* source_batch =
          plan.input_size == 0
              ? source_data
              : source_data + batch * plan.input_size;
      auto* result_batch =
          plan.output_size == 0
              ? result_data
              : result_data + batch * plan.output_size;
      for (const ReductionRecord& record : plan.records) {
        ExecuteReductionOutputRange(
            record, source_batch, result_batch, 0, record.output_count,
            scalar_op, add_op);
      }
    }
    return ReadyFuture();
  };

  if (plan.records.size() != 1 || batch_count == 0 ||
      available_workers <= 1) {
    return execute_sequential();
  }
  const ReductionRecord& record = plan.records.front();
  if (record.output_count == 0 || record.reduction_count == 0) {
    return execute_sequential();
  }
  uint64_t fiber_bytes = 0;
  uint64_t total_fibers = 0;
  uint64_t total_work_bytes = 0;
  if (!CheckedMultiply(record.reduction_count, source_item_size,
                       &fiber_bytes) ||
      !CheckedMultiply(record.output_count, batch_count, &total_fibers) ||
      !CheckedMultiply(total_fibers, fiber_bytes, &total_work_bytes)) {
    return ErrorFuture(Invalid("reduction parallel work size overflow"));
  }
  if (total_work_bytes < plan.parallel_minimum_bytes) {
    return execute_sequential();
  }
  const uint64_t outputs_per_chunk = std::max<uint64_t>(
      1, fiber_bytes == 0 ? record.output_count
                          : plan.preferred_chunk_bytes / fiber_bytes);
  const uint64_t chunks_per_batch =
      1 + (record.output_count - 1) / outputs_per_chunk;
  uint64_t total_chunks = 0;
  if (!CheckedMultiply(chunks_per_batch, batch_count, &total_chunks)) {
    return ErrorFuture(Invalid("reduction parallel chunk count overflow"));
  }
  const uint64_t worker_count = std::min(available_workers, total_chunks);
  if (worker_count <= 1) {
    return execute_sequential();
  }

  using SourceElement =
      std::remove_const_t<std::remove_pointer_t<decltype(source_data)>>;
  using State = ReductionParallelState<SourceElement, ResultElement,
                                       ScalarOp, AddOp>;
  auto state = std::make_shared<State>();
  state->execution = execution;
  state->source = source_data;
  state->result = result_data;
  state->batch_count = batch_count;
  state->chunks_per_batch = chunks_per_batch;
  state->total_chunks = total_chunks;
  state->outputs_per_chunk = outputs_per_chunk;
  state->scalar_op = scalar_op;
  state->add_op = add_op;
  last_worker_count.store(worker_count, std::memory_order_relaxed);
  ffi::CountDownPromise completion(static_cast<int64_t>(worker_count));
  ffi::Future future(completion);
  uint64_t scheduled = 0;
  for (; scheduled < worker_count; ++scheduled) {
    try {
      thread_pool.Schedule([state, completion]() mutable {
        try {
          ExecuteReductionParallelWorker(state);
          completion.CountDown();
        } catch (const std::exception&) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained reduction worker exception"));
        } catch (...) {
          completion.CountDown(ffi::Error::Internal(
              "tensor0-stride: contained unknown reduction worker exception"));
        }
      });
    } catch (const std::exception&) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: failed to schedule reduction worker"));
      break;
    } catch (...) {
      completion.CountDown(
          static_cast<std::size_t>(worker_count - scheduled),
          ffi::Error::Internal(
              "tensor0-stride: unknown reduction worker scheduling failure"));
      break;
    }
  }
  return future;
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
                       ReductionScaleF32, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16Impl, ffi::F16,
                       ffi::F16, kDtypeF16, kDtypeF16,
                       kDtypeF16, kReductionScalarJaxTranspose,
                       ReductionScaleF16, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionBF16Impl, ffi::BF16,
                       ffi::BF16, kDtypeBF16, kDtypeBF16,
                       kDtypeBF16, kReductionScalarJaxTranspose,
                       ReductionScaleBF16, AddBF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64Impl, ffi::C64,
                       ffi::C64, kDtypeC64, kDtypeC64,
                       kDtypeC64, kReductionScalarJaxTranspose,
                       ReductionScaleC64, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32F16Impl, ffi::F32,
                       ffi::F16, kDtypeF32, kDtypeF16,
                       kDtypeF32, kReductionScalarJaxTranspose,
                       ReductionScaleF32F16Transpose, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64F32Impl, ffi::C64,
                       ffi::F32, kDtypeC64, kDtypeF32,
                       kDtypeC64, kReductionScalarJaxTranspose,
                       ReductionScaleC64F32Transpose, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32C64Impl, ffi::F32,
                       ffi::C64, kDtypeF32, kDtypeC64,
                       kDtypeF32, kReductionScalarJaxTranspose,
                       ReductionScaleF32C64Transpose, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64Impl, ffi::F64,
                       ffi::F64, kDtypeF64, kDtypeF64,
                       kDtypeF64, kReductionScalarJaxTranspose,
                       ReductionScaleF64, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128Impl, ffi::C128,
                       ffi::C128, kDtypeC128, kDtypeC128,
                       kDtypeC128, kReductionScalarJaxTranspose,
                       ReductionScaleC128, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128F64Impl, ffi::C128,
                       ffi::F64, kDtypeC128, kDtypeF64,
                       kDtypeC128, kReductionScalarJaxTranspose,
                       ReductionScaleC128F64Transpose, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64C128Impl, ffi::F64,
                       ffi::C128, kDtypeF64, kDtypeC128,
                       kDtypeF64, kReductionScalarJaxTranspose,
                       ReductionScaleF64C128Transpose, AddC128)

TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32ForwardImpl, ffi::F32,
                       ffi::F32, kDtypeF32, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       ReductionScaleF32, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16ForwardImpl, ffi::F16,
                       ffi::F16, kDtypeF16, kDtypeF16,
                       kDtypeF16, kReductionScalarForwardScaleCast,
                       ReductionScaleF16, AddF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionBF16ForwardImpl, ffi::BF16,
                       ffi::BF16, kDtypeBF16, kDtypeBF16,
                       kDtypeBF16, kReductionScalarForwardScaleCast,
                       ReductionScaleBF16, AddBF16)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64ForwardImpl, ffi::C64,
                       ffi::C64, kDtypeC64, kDtypeC64,
                       kDtypeC64, kReductionScalarForwardScaleCast,
                       ReductionScaleC64, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionPredForwardImpl, ffi::PRED,
                       ffi::PRED, kDtypePred, kDtypePred,
                       kDtypePred, kReductionScalarForwardScaleCast,
                       ReductionScalePred, AddPred)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS8ForwardImpl, ffi::S8,
                       ffi::S8, kDtypeS8, kDtypeS8,
                       kDtypeS8, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<int8_t>, AddIntegerOp<int8_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS16ForwardImpl, ffi::S16,
                       ffi::S16, kDtypeS16, kDtypeS16,
                       kDtypeS16, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<int16_t>, AddIntegerOp<int16_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS32ForwardImpl, ffi::S32,
                       ffi::S32, kDtypeS32, kDtypeS32,
                       kDtypeS32, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<int32_t>, AddIntegerOp<int32_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionS64ForwardImpl, ffi::S64,
                       ffi::S64, kDtypeS64, kDtypeS64,
                       kDtypeS64, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<int64_t>, AddIntegerOp<int64_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU8ForwardImpl, ffi::U8,
                       ffi::U8, kDtypeU8, kDtypeU8,
                       kDtypeU8, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<uint8_t>, AddIntegerOp<uint8_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU16ForwardImpl, ffi::U16,
                       ffi::U16, kDtypeU16, kDtypeU16,
                       kDtypeU16, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<uint16_t>, AddIntegerOp<uint16_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU32ForwardImpl, ffi::U32,
                       ffi::U32, kDtypeU32, kDtypeU32,
                       kDtypeU32, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<uint32_t>, AddIntegerOp<uint32_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionU64ForwardImpl, ffi::U64,
                       ffi::U64, kDtypeU64, kDtypeU64,
                       kDtypeU64, kReductionScalarForwardScaleCast,
                       ReductionScaleInteger<uint64_t>, AddIntegerOp<uint64_t>)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF16F32ForwardImpl, ffi::F16,
                       ffi::F32, kDtypeF16, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       ReductionScaleF16F32Forward, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF32C64ForwardImpl, ffi::F32,
                       ffi::C64, kDtypeF32, kDtypeC64,
                       kDtypeC64, kReductionScalarForwardScaleCast,
                       ReductionScaleF32C64Forward, AddC64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC64F32ForwardImpl, ffi::C64,
                       ffi::F32, kDtypeC64, kDtypeF32,
                       kDtypeF32, kReductionScalarForwardScaleCast,
                       ReductionScaleC64F32Forward, AddF32)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64ForwardImpl, ffi::F64,
                       ffi::F64, kDtypeF64, kDtypeF64,
                       kDtypeF64, kReductionScalarForwardScaleCast,
                       ReductionScaleF64, AddF64)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128ForwardImpl, ffi::C128,
                       ffi::C128, kDtypeC128, kDtypeC128,
                       kDtypeC128, kReductionScalarForwardScaleCast,
                       ReductionScaleC128, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionF64C128ForwardImpl, ffi::F64,
                       ffi::C128, kDtypeF64, kDtypeC128,
                       kDtypeC128, kReductionScalarForwardScaleCast,
                       ReductionScaleF64C128Forward, AddC128)
TENSOR0_REDUCTION_IMPL(ExecuteStructuredReductionC128F64ForwardImpl, ffi::C128,
                       ffi::F64, kDtypeC128, kDtypeF64,
                       kDtypeF64, kReductionScalarForwardScaleCast,
                       ReductionScaleC128F64Forward, AddF64)

#undef TENSOR0_REDUCTION_IMPL

}  // namespace
}  // namespace tensor0::stride


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedF32V7Execute,
    tensor0::stride::ExecutePreparedF32Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedF16V7Execute,
    tensor0::stride::ExecutePreparedF16Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::F16>>()
        .Ret<ffi::Buffer<ffi::F16>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedBF16V7Execute,
    tensor0::stride::ExecutePreparedBF16Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::BF16>>()
        .Ret<ffi::Buffer<ffi::BF16>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedC64V7Execute,
    tensor0::stride::ExecutePreparedC64Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::C64>>()
        .Ret<ffi::Buffer<ffi::C64>>()
        .Ctx<ffi::ThreadPool>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedS32V7Execute,
    tensor0::stride::ExecutePreparedS32Impl,
    ffi::Ffi::BindExecute()
        .Attr<ffi::Span<const uint8_t>>("descriptor")
        .Ctx<ffi::State<tensor0::stride::PreparedState>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ctx<ffi::ThreadPool>());

#define TENSOR0_DEFINE_PREPARED_EXECUTE(Symbol, Function, Dtype)             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Ret<ffi::Buffer<Dtype>>()                                         \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedPredV7Execute,
                                tensor0::stride::ExecutePreparedPredImpl,
                                ffi::PRED)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedS8V7Execute,
                                tensor0::stride::ExecutePreparedS8Impl,
                                ffi::S8)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedS16V7Execute,
                                tensor0::stride::ExecutePreparedS16Impl,
                                ffi::S16)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedS64V7Execute,
                                tensor0::stride::ExecutePreparedS64Impl,
                                ffi::S64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedU8V7Execute,
                                tensor0::stride::ExecutePreparedU8Impl,
                                ffi::U8)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedU16V7Execute,
                                tensor0::stride::ExecutePreparedU16Impl,
                                ffi::U16)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedU32V7Execute,
                                tensor0::stride::ExecutePreparedU32Impl,
                                ffi::U32)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedU64V7Execute,
                                tensor0::stride::ExecutePreparedU64Impl,
                                ffi::U64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedF64V7Execute,
                                tensor0::stride::ExecutePreparedF64Impl,
                                ffi::F64)
TENSOR0_DEFINE_PREPARED_EXECUTE(Tensor0StrideR2PreparedC128V7Execute,
                                tensor0::stride::ExecutePreparedC128Impl,
                                ffi::C128)

#undef TENSOR0_DEFINE_PREPARED_EXECUTE

#define TENSOR0_DEFINE_MIXED_EXECUTE(Symbol, Function, SourceType, ResultType) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                           \
      Symbol, Function,                                                    \
      ffi::Ffi::BindExecute()                                              \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                    \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()               \
          .Arg<ffi::Buffer<SourceType>>()                                  \
          .Ret<ffi::Buffer<ResultType>>()                                  \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF16F32ForwardV1Execute,
    tensor0::stride::ExecutePreparedF16F32ForwardImpl, ffi::F16, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32F16TransposeV1Execute,
    tensor0::stride::ExecutePreparedF32F16TransposeImpl, ffi::F32, ffi::F16)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32C64ForwardV1Execute,
    tensor0::stride::ExecutePreparedF32C64ForwardImpl, ffi::F32, ffi::C64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC64F32TransposeV1Execute,
    tensor0::stride::ExecutePreparedC64F32TransposeImpl, ffi::C64, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC64F32ForwardV1Execute,
    tensor0::stride::ExecutePreparedC64F32ForwardImpl, ffi::C64, ffi::F32)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF32C64TransposeV1Execute,
    tensor0::stride::ExecutePreparedF32C64TransposeImpl, ffi::F32, ffi::C64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF64C128ForwardV1Execute,
    tensor0::stride::ExecutePreparedF64C128ForwardImpl, ffi::F64, ffi::C128)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC128F64TransposeV1Execute,
    tensor0::stride::ExecutePreparedC128F64TransposeImpl, ffi::C128, ffi::F64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineC128F64ForwardV1Execute,
    tensor0::stride::ExecutePreparedC128F64ForwardImpl, ffi::C128, ffi::F64)
TENSOR0_DEFINE_MIXED_EXECUTE(
    Tensor0StrideAffineF64C128TransposeV1Execute,
    tensor0::stride::ExecutePreparedF64C128TransposeImpl, ffi::F64, ffi::C128)

#undef TENSOR0_DEFINE_MIXED_EXECUTE

#define TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(Symbol, Function, SourceType,     \
                                           ResultType)                       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<ResultType>>()                                    \
          .Arg<ffi::Buffer<SourceType>>()                                    \
          .Ret<ffi::Buffer<ResultType>>()                                    \
          .Ctx<ffi::ThreadPool>());

#define TENSOR0_DEFINE_BASE_UPDATE_DTYPE_SYMBOLS(Prefix)                      \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F32V1Execute, tensor0::stride::Prefix##F32Impl, ffi::F32,       \
      ffi::F32)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F16V1Execute, tensor0::stride::Prefix##F16Impl, ffi::F16,       \
      ffi::F16)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##BF16V1Execute, tensor0::stride::Prefix##BF16Impl, ffi::BF16,    \
      ffi::BF16)                                                              \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##C64V1Execute, tensor0::stride::Prefix##C64Impl, ffi::C64,       \
      ffi::C64)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S32V1Execute, tensor0::stride::Prefix##S32Impl, ffi::S32,       \
      ffi::S32)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##PredV1Execute, tensor0::stride::Prefix##PredImpl, ffi::PRED,    \
      ffi::PRED)                                                              \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S8V1Execute, tensor0::stride::Prefix##S8Impl, ffi::S8, ffi::S8) \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S16V1Execute, tensor0::stride::Prefix##S16Impl, ffi::S16,       \
      ffi::S16)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S64V1Execute, tensor0::stride::Prefix##S64Impl, ffi::S64,       \
      ffi::S64)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##U8V1Execute, tensor0::stride::Prefix##U8Impl, ffi::U8, ffi::U8) \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##U16V1Execute, tensor0::stride::Prefix##U16Impl, ffi::U16,       \
      ffi::U16)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##U32V1Execute, tensor0::stride::Prefix##U32Impl, ffi::U32,       \
      ffi::U32)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##U64V1Execute, tensor0::stride::Prefix##U64Impl, ffi::U64,       \
      ffi::U64)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F64V1Execute, tensor0::stride::Prefix##F64Impl, ffi::F64,       \
      ffi::F64)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##C128V1Execute, tensor0::stride::Prefix##C128Impl, ffi::C128,    \
      ffi::C128)                                                              \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F16F32V1Execute, tensor0::stride::Prefix##F16F32Impl, ffi::F16, \
      ffi::F32)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F32C64V1Execute, tensor0::stride::Prefix##F32C64Impl, ffi::F32, \
      ffi::C64)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##C64F32V1Execute, tensor0::stride::Prefix##C64F32Impl, ffi::C64, \
      ffi::F32)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##F64C128V1Execute, tensor0::stride::Prefix##F64C128Impl,         \
      ffi::F64, ffi::C128)                                                    \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##C128F64V1Execute, tensor0::stride::Prefix##C128F64Impl,         \
      ffi::C128, ffi::F64)                                                    \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S32PredV1Execute, tensor0::stride::Prefix##S32PredImpl,         \
      ffi::S32, ffi::PRED)                                                    \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S32S8V1Execute, tensor0::stride::Prefix##S32S8Impl, ffi::S32,   \
      ffi::S8)                                                               \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##S32S16V1Execute, tensor0::stride::Prefix##S32S16Impl,           \
      ffi::S32, ffi::S16)                                                     \
  TENSOR0_DEFINE_BASE_UPDATE_EXECUTE(                                        \
      Prefix##U32U8V1Execute, tensor0::stride::Prefix##U32U8Impl, ffi::U32,   \
      ffi::U8)

TENSOR0_DEFINE_BASE_UPDATE_DTYPE_SYMBOLS(ExecuteBaseAssign)
TENSOR0_DEFINE_BASE_UPDATE_DTYPE_SYMBOLS(ExecuteBaseAccumulate)

#undef TENSOR0_DEFINE_BASE_UPDATE_DTYPE_SYMBOLS
#undef TENSOR0_DEFINE_BASE_UPDATE_EXECUTE

#define TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(Symbol, Function, Dtype)       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Arg<ffi::Buffer<Dtype>>()                                         \
          .Ret<ffi::Buffer<Dtype>>()                                         \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleF32V1Execute,
    tensor0::stride::ExecuteSelectedScaleF32Impl, ffi::F32)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleF16V1Execute,
    tensor0::stride::ExecuteSelectedScaleF16Impl, ffi::F16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleBF16V1Execute,
    tensor0::stride::ExecuteSelectedScaleBF16Impl, ffi::BF16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleC64V1Execute,
    tensor0::stride::ExecuteSelectedScaleC64Impl, ffi::C64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleS32V1Execute,
    tensor0::stride::ExecuteSelectedScaleS32Impl, ffi::S32)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScalePredV1Execute,
    tensor0::stride::ExecuteSelectedScalePredImpl, ffi::PRED)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleS8V1Execute,
    tensor0::stride::ExecuteSelectedScaleS8Impl, ffi::S8)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleS16V1Execute,
    tensor0::stride::ExecuteSelectedScaleS16Impl, ffi::S16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleS64V1Execute,
    tensor0::stride::ExecuteSelectedScaleS64Impl, ffi::S64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleU8V1Execute,
    tensor0::stride::ExecuteSelectedScaleU8Impl, ffi::U8)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleU16V1Execute,
    tensor0::stride::ExecuteSelectedScaleU16Impl, ffi::U16)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleU32V1Execute,
    tensor0::stride::ExecuteSelectedScaleU32Impl, ffi::U32)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleU64V1Execute,
    tensor0::stride::ExecuteSelectedScaleU64Impl, ffi::U64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleF64V1Execute,
    tensor0::stride::ExecuteSelectedScaleF64Impl, ffi::F64)
TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE(
    Tensor0StrideSelectedScaleC128V1Execute,
    tensor0::stride::ExecuteSelectedScaleC128Impl, ffi::C128)

#undef TENSOR0_DEFINE_SELECTED_SCALE_EXECUTE

#define TENSOR0_DEFINE_REDUCTION_EXECUTE(Symbol, Function, SourceType,       \
                                         ResultType)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                             \
      Symbol, Function,                                                      \
      ffi::Ffi::BindExecute()                                                \
          .Attr<ffi::Span<const uint8_t>>("descriptor")                     \
          .Ctx<ffi::State<tensor0::stride::PreparedState>>()                 \
          .Arg<ffi::Buffer<SourceType>>()                                    \
          .Ret<ffi::Buffer<ResultType>>()                                    \
          .Ctx<ffi::ThreadPool>());

TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32Impl, ffi::F32, ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16V1Execute,
    tensor0::stride::ExecuteStructuredReductionF16Impl, ffi::F16, ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionBF16V1Execute,
    tensor0::stride::ExecuteStructuredReductionBF16Impl, ffi::BF16,
    ffi::BF16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64V1Execute,
    tensor0::stride::ExecuteStructuredReductionC64Impl, ffi::C64, ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32F16V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32F16Impl, ffi::F32,
    ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64F32V1Execute,
    tensor0::stride::ExecuteStructuredReductionC64F32Impl, ffi::C64,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32C64V1Execute,
    tensor0::stride::ExecuteStructuredReductionF32C64Impl, ffi::F32,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64V1Execute,
    tensor0::stride::ExecuteStructuredReductionF64Impl, ffi::F64, ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128V1Execute,
    tensor0::stride::ExecuteStructuredReductionC128Impl, ffi::C128,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128F64V1Execute,
    tensor0::stride::ExecuteStructuredReductionC128F64Impl, ffi::C128,
    ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64C128V1Execute,
    tensor0::stride::ExecuteStructuredReductionF64C128Impl, ffi::F64,
    ffi::C128)

TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF32ForwardImpl, ffi::F32,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF16ForwardImpl, ffi::F16,
    ffi::F16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionBF16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionBF16ForwardImpl, ffi::BF16,
    ffi::BF16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC64ForwardImpl, ffi::C64,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionPredForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionPredForwardImpl, ffi::PRED,
    ffi::PRED)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS8ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS8ForwardImpl, ffi::S8,
    ffi::S8)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS16ForwardImpl, ffi::S16,
    ffi::S16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS32ForwardImpl, ffi::S32,
    ffi::S32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionS64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionS64ForwardImpl, ffi::S64,
    ffi::S64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU8ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU8ForwardImpl, ffi::U8,
    ffi::U8)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU16ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU16ForwardImpl, ffi::U16,
    ffi::U16)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU32ForwardImpl, ffi::U32,
    ffi::U32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionU64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionU64ForwardImpl, ffi::U64,
    ffi::U64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF16F32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF16F32ForwardImpl, ffi::F16,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF32C64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF32C64ForwardImpl, ffi::F32,
    ffi::C64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC64F32ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC64F32ForwardImpl, ffi::C64,
    ffi::F32)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF64ForwardImpl, ffi::F64,
    ffi::F64)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC128ForwardImpl, ffi::C128,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionF64C128ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionF64C128ForwardImpl, ffi::F64,
    ffi::C128)
TENSOR0_DEFINE_REDUCTION_EXECUTE(
    Tensor0StrideStructuredReductionC128F64ForwardV2Execute,
    tensor0::stride::ExecuteStructuredReductionC128F64ForwardImpl, ffi::C128,
    ffi::F64)

#undef TENSOR0_DEFINE_REDUCTION_EXECUTE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideR2PreparedF32V7Instantiate,
    tensor0::stride::InstantiatePreparedImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideBaseAssignV1Instantiate,
    tensor0::stride::InstantiatePreparedOperationImpl<
        tensor0::stride::PreparedOperation::kBaseAssign>,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideBaseAccumulateV1Instantiate,
    tensor0::stride::InstantiatePreparedOperationImpl<
        tensor0::stride::PreparedOperation::kBaseAccumulate>,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideSelectedScaleV1Instantiate,
    tensor0::stride::InstantiatePreparedOperationImpl<
        tensor0::stride::PreparedOperation::kSelectedScale>,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideStructuredReductionV1Instantiate,
    tensor0::stride::InstantiatePreparedReductionImpl,
    ffi::Ffi::BindInstantiate()
        .Attr<ffi::Span<const uint8_t>>("descriptor"));

extern "C" void* Tensor0StrideR2PreparedF32V7InstantiateHandler() {
  return reinterpret_cast<void*>(
      &Tensor0StrideR2PreparedF32V7Instantiate);
}

extern "C" void* Tensor0StrideR2PreparedF32V7ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideR2PreparedF32V7Execute);
}

extern "C" void* Tensor0StrideR2PreparedF16V7ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideR2PreparedF16V7Execute);
}

extern "C" void* Tensor0StrideR2PreparedBF16V7ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideR2PreparedBF16V7Execute);
}

extern "C" void* Tensor0StrideR2PreparedC64V7ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideR2PreparedC64V7Execute);
}

extern "C" void* Tensor0StrideR2PreparedS32V7ExecuteHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideR2PreparedS32V7Execute);
}

#define TENSOR0_PREPARED_HANDLER(Name, Symbol) \
  extern "C" void* Name() { return reinterpret_cast<void*>(&Symbol); }

TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedPredV7ExecuteHandler,
                         Tensor0StrideR2PreparedPredV7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedS8V7ExecuteHandler,
                         Tensor0StrideR2PreparedS8V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedS16V7ExecuteHandler,
                         Tensor0StrideR2PreparedS16V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedS64V7ExecuteHandler,
                         Tensor0StrideR2PreparedS64V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedU8V7ExecuteHandler,
                         Tensor0StrideR2PreparedU8V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedU16V7ExecuteHandler,
                         Tensor0StrideR2PreparedU16V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedU32V7ExecuteHandler,
                         Tensor0StrideR2PreparedU32V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedU64V7ExecuteHandler,
                         Tensor0StrideR2PreparedU64V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedF64V7ExecuteHandler,
                         Tensor0StrideR2PreparedF64V7Execute)
TENSOR0_PREPARED_HANDLER(Tensor0StrideR2PreparedC128V7ExecuteHandler,
                         Tensor0StrideR2PreparedC128V7Execute)

#undef TENSOR0_PREPARED_HANDLER

#define TENSOR0_MIXED_HANDLER(Name, Symbol)       \
  extern "C" void* Name() {                      \
    return reinterpret_cast<void*>(&Symbol);     \
  }

TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF16F32ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF16F32ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32F16TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF32F16TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32C64ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF32C64ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC64F32TransposeV1ExecuteHandler,
                      Tensor0StrideAffineC64F32TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC64F32ForwardV1ExecuteHandler,
                      Tensor0StrideAffineC64F32ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF32C64TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF32C64TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF64C128ForwardV1ExecuteHandler,
                      Tensor0StrideAffineF64C128ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC128F64TransposeV1ExecuteHandler,
                      Tensor0StrideAffineC128F64TransposeV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineC128F64ForwardV1ExecuteHandler,
                      Tensor0StrideAffineC128F64ForwardV1Execute)
TENSOR0_MIXED_HANDLER(Tensor0StrideAffineF64C128TransposeV1ExecuteHandler,
                      Tensor0StrideAffineF64C128TransposeV1Execute)

#undef TENSOR0_MIXED_HANDLER

extern "C" void* Tensor0StrideBaseAssignV1InstantiateHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideBaseAssignV1Instantiate);
}

extern "C" void* Tensor0StrideBaseAccumulateV1InstantiateHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideBaseAccumulateV1Instantiate);
}

extern "C" void* Tensor0StrideSelectedScaleV1InstantiateHandler() {
  return reinterpret_cast<void*>(&Tensor0StrideSelectedScaleV1Instantiate);
}

#define TENSOR0_BASE_UPDATE_HANDLER(Name, Symbol) \
  extern "C" void* Name() {                     \
    return reinterpret_cast<void*>(&Symbol);     \
  }

#define TENSOR0_BASE_UPDATE_DTYPE_HANDLERS(Prefix)                             \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##F32V1ExecuteHandler,     \
                              Execute##Prefix##F32V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##F16V1ExecuteHandler,     \
                              Execute##Prefix##F16V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##BF16V1ExecuteHandler,    \
                              Execute##Prefix##BF16V1Execute)                 \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##C64V1ExecuteHandler,     \
                              Execute##Prefix##C64V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##S32V1ExecuteHandler,     \
                              Execute##Prefix##S32V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##PredV1ExecuteHandler,    \
                              Execute##Prefix##PredV1Execute)                 \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##S8V1ExecuteHandler,      \
                              Execute##Prefix##S8V1Execute)                   \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##S16V1ExecuteHandler,     \
                              Execute##Prefix##S16V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##S64V1ExecuteHandler,     \
                              Execute##Prefix##S64V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##U8V1ExecuteHandler,      \
                              Execute##Prefix##U8V1Execute)                   \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##U16V1ExecuteHandler,     \
                              Execute##Prefix##U16V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##U32V1ExecuteHandler,     \
                              Execute##Prefix##U32V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##U64V1ExecuteHandler,     \
                              Execute##Prefix##U64V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##F64V1ExecuteHandler,     \
                              Execute##Prefix##F64V1Execute)                  \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##C128V1ExecuteHandler,    \
                              Execute##Prefix##C128V1Execute)                 \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##F16F32V1ExecuteHandler,  \
                              Execute##Prefix##F16F32V1Execute)               \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##F32C64V1ExecuteHandler,  \
                              Execute##Prefix##F32C64V1Execute)               \
  TENSOR0_BASE_UPDATE_HANDLER(Tensor0Stride##Prefix##C64F32V1ExecuteHandler,  \
                              Execute##Prefix##C64F32V1Execute)               \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##F64C128V1ExecuteHandler,                         \
      Execute##Prefix##F64C128V1Execute)                                      \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##C128F64V1ExecuteHandler,                         \
      Execute##Prefix##C128F64V1Execute)                                      \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##S32PredV1ExecuteHandler,                         \
      Execute##Prefix##S32PredV1Execute)                                      \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##S32S8V1ExecuteHandler,                           \
      Execute##Prefix##S32S8V1Execute)                                        \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##S32S16V1ExecuteHandler,                          \
      Execute##Prefix##S32S16V1Execute)                                       \
  TENSOR0_BASE_UPDATE_HANDLER(                                                \
      Tensor0Stride##Prefix##U32U8V1ExecuteHandler,                           \
      Execute##Prefix##U32U8V1Execute)

TENSOR0_BASE_UPDATE_DTYPE_HANDLERS(BaseAssign)
TENSOR0_BASE_UPDATE_DTYPE_HANDLERS(BaseAccumulate)

#undef TENSOR0_BASE_UPDATE_DTYPE_HANDLERS

TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleF32V1ExecuteHandler,
                            Tensor0StrideSelectedScaleF32V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleF16V1ExecuteHandler,
                            Tensor0StrideSelectedScaleF16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleBF16V1ExecuteHandler,
                            Tensor0StrideSelectedScaleBF16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleC64V1ExecuteHandler,
                            Tensor0StrideSelectedScaleC64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleS32V1ExecuteHandler,
                            Tensor0StrideSelectedScaleS32V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScalePredV1ExecuteHandler,
                            Tensor0StrideSelectedScalePredV1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleS8V1ExecuteHandler,
                            Tensor0StrideSelectedScaleS8V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleS16V1ExecuteHandler,
                            Tensor0StrideSelectedScaleS16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleS64V1ExecuteHandler,
                            Tensor0StrideSelectedScaleS64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleU8V1ExecuteHandler,
                            Tensor0StrideSelectedScaleU8V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleU16V1ExecuteHandler,
                            Tensor0StrideSelectedScaleU16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleU32V1ExecuteHandler,
                            Tensor0StrideSelectedScaleU32V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleU64V1ExecuteHandler,
                            Tensor0StrideSelectedScaleU64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleF64V1ExecuteHandler,
                            Tensor0StrideSelectedScaleF64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(Tensor0StrideSelectedScaleC128V1ExecuteHandler,
                            Tensor0StrideSelectedScaleC128V1Execute)

TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionV1InstantiateHandler,
    Tensor0StrideStructuredReductionV1Instantiate)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF32V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF16V1ExecuteHandler,
    Tensor0StrideStructuredReductionF16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionBF16V1ExecuteHandler,
    Tensor0StrideStructuredReductionBF16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC64V1ExecuteHandler,
    Tensor0StrideStructuredReductionC64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF32F16V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32F16V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC64F32V1ExecuteHandler,
    Tensor0StrideStructuredReductionC64F32V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF32C64V1ExecuteHandler,
    Tensor0StrideStructuredReductionF32C64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF64V1ExecuteHandler,
    Tensor0StrideStructuredReductionF64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC128V1ExecuteHandler,
    Tensor0StrideStructuredReductionC128V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC128F64V1ExecuteHandler,
    Tensor0StrideStructuredReductionC128F64V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF64C128V1ExecuteHandler,
    Tensor0StrideStructuredReductionF64C128V1Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF32ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF16ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionBF16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionBF16ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC64ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionPredForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionPredForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionS8ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS8ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionS16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS16ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionS32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS32ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionS64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionS64ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionU8ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU8ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionU16ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU16ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionU32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU32ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionU64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionU64ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF16F32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF16F32ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF32C64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF32C64ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC64F32ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC64F32ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF64ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC128ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC128ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionF64C128ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionF64C128ForwardV2Execute)
TENSOR0_BASE_UPDATE_HANDLER(
    Tensor0StrideStructuredReductionC128F64ForwardV2ExecuteHandler,
    Tensor0StrideStructuredReductionC128F64ForwardV2Execute)

#undef TENSOR0_BASE_UPDATE_HANDLER

extern "C" void* Tensor0StridePreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::PreparedState::id);
}

extern "C" const void* Tensor0StridePreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::kPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StridePreparedInstantiateCount() {
  return tensor0::stride::prepared_instantiate_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedExecuteCount() {
  return tensor0::stride::prepared_execute_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLiveStateCount() {
  return tensor0::stride::prepared_live_state_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedDestroyedStateCount() {
  return tensor0::stride::prepared_destroyed_state_count.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLiveBytes() {
  return tensor0::stride::prepared_live_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLastStateBytes() {
  return tensor0::stride::prepared_last_state_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedLastDescriptorBytes() {
  return tensor0::stride::prepared_last_descriptor_bytes.load(
      std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedResetMetrics() {
  if (tensor0::stride::prepared_live_state_count.load(
          std::memory_order_relaxed) != 0) {
    return 0;
  }
  tensor0::stride::prepared_instantiate_count.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_execute_count.store(0,
                                                std::memory_order_relaxed);
  tensor0::stride::prepared_destroyed_state_count.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_live_bytes.store(0,
                                             std::memory_order_relaxed);
  tensor0::stride::prepared_last_state_bytes.store(
      0, std::memory_order_relaxed);
  tensor0::stride::prepared_last_descriptor_bytes.store(
      0, std::memory_order_relaxed);
  return 1;
}

extern "C" uint64_t Tensor0StrideNativeCallCount() {
  return tensor0::stride::native_call_count.load(std::memory_order_relaxed);
}

extern "C" void Tensor0StrideResetNativeCallCount() {
  tensor0::stride::native_call_count.store(0, std::memory_order_relaxed);
  tensor0::stride::last_worker_count.store(0, std::memory_order_relaxed);
  tensor0::stride::last_available_worker_count.store(
      0, std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastWorkerCount() {
  return tensor0::stride::last_worker_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StrideLastAvailableWorkerCount() {
  return tensor0::stride::last_available_worker_count.load(
      std::memory_order_relaxed);
}

extern "C" void Tensor0StrideSetWorkerLimitForTests(uint64_t limit) {
  tensor0::stride::worker_limit.store(limit, std::memory_order_relaxed);
}

extern "C" const char* Tensor0StrideBuiltJaxVersion() {
  return TENSOR0_STRIDE_JAX_VERSION;
}

extern "C" const char* Tensor0StrideBuiltJaxlibVersion() {
  return TENSOR0_STRIDE_JAXLIB_VERSION;
}

extern "C" uint64_t Tensor0StrideAbiVersion() {
  return tensor0::stride::kCompiledDescriptorVersion;
}
