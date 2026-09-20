#include "prepared.h"

#include "errors.h"
#include "../layout/traversal.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tensor0::stride {

namespace descriptor {

DecodedLayout DecodeAddressLayout(const int64_t* words, std::size_t word_count) {
  std::size_t cursor = 0;
  const auto read = [&]() {
    if (cursor == word_count) {
      throw std::invalid_argument("layout descriptor is truncated");
    }
    return words[cursor++];
  };
  const auto read_size = [&]() {
    const auto value = read();
    if (value < 0) {
      throw std::invalid_argument("layout size or offset is negative");
    }
    return static_cast<uint64_t>(value);
  };
  if (read() != kLayoutVersion) {
    throw std::invalid_argument("unsupported native layout version");
  }
  DecodedLayout decoded;
  auto& records = decoded.records;
  const auto source_size = decoded.source_size = read_size();
  const auto output_size = decoded.output_size = read_size();
  const auto record_count = read_size();
  if (record_count > (word_count - cursor) / 3) {
    throw std::invalid_argument("layout record count exceeds descriptor length");
  }
  records.reserve(record_count);
  for (std::size_t index = 0; index < record_count; ++index) {
    const auto rank = read_size();
    const auto source_offset = static_cast<int64_t>(read_size());
    const auto destination_offset = static_cast<int64_t>(read_size());
    if (rank > (word_count - cursor) / 3) {
      throw std::invalid_argument("layout rank exceeds descriptor length");
    }
    std::vector<uint64_t> shape(rank);
    std::vector<int64_t> source_strides(rank), destination_strides(rank);
    for (auto& extent : shape) {
      extent = read_size();
    }
    for (auto& stride : source_strides) stride = read();
    for (auto& stride : destination_strides) stride = read();
    auto record = layout::BuildLayout(
        shape, source_strides, source_offset, destination_strides, destination_offset,
        source_size, output_size, index);
    records.push_back(std::move(record));
  }
  if (cursor != word_count) {
    throw std::invalid_argument("layout descriptor has trailing words");
  }
  return decoded;
}

DecodedLayout DecodeLayout(const int64_t* words, std::size_t word_count) {
  auto decoded = DecodeAddressLayout(words, word_count);
  for (auto& record : decoded.records) {
    const auto& shape = record.shape;
    std::vector<std::size_t> axes(std::find(shape.begin(), shape.end(), 0) == shape.end() ? shape.size() : 0);
    std::iota(axes.begin(), axes.end(), 0);
    layout::ValidateInjectiveView(record, false, axes);
    layout::OptimizeRecordForExecution(&record);
  }
  return decoded;
}

DecodedLayout DecodeReductionLayout(const char* bytes, std::size_t byte_count) {
  if (byte_count % sizeof(uint64_t) != 0) {
    throw std::invalid_argument("reduction layout descriptor has a partial word");
  }
  std::size_t cursor = 0;
  const auto read = [&]() {
    if (cursor == byte_count) {
      throw std::invalid_argument("reduction layout descriptor is truncated");
    }
    uint64_t word = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      word |= static_cast<uint64_t>(static_cast<uint8_t>(bytes[cursor++])) << shift;
    }
    return word;
  };
  const auto read_stride = [&]() {
    const auto word = read();
    return word <= static_cast<uint64_t>(INT64_MAX)
        ? static_cast<int64_t>(word) : -1 - static_cast<int64_t>(UINT64_MAX - word);
  };
  const auto read_offset = [&]() {
    const auto word = read();
    if (word > static_cast<uint64_t>(INT64_MAX)) {
      throw std::invalid_argument("reduction layout offset exceeds address range");
    }
    return static_cast<int64_t>(word);
  };
  if (read() != kReductionLayoutVersion) {
    throw std::invalid_argument("unsupported native reduction layout version");
  }
  DecodedLayout decoded;
  decoded.source_size = read();
  decoded.output_size = read();
  const auto record_count = read();
  if (record_count > (byte_count - cursor) / (3 * sizeof(uint64_t))) {
    throw std::invalid_argument("reduction layout record count exceeds descriptor length");
  }
  decoded.records.reserve(record_count);
  for (std::size_t index = 0; index < record_count; ++index) {
    const auto rank = read();
    const auto source_offset = read_offset();
    const auto output_offset = read_offset();
    if (rank > (byte_count - cursor) / (5 * sizeof(uint64_t))) {
      throw std::invalid_argument("reduction layout rank exceeds descriptor length");
    }
    std::vector<uint64_t> source_shape(rank), output_shape(rank);
    std::vector<int64_t> source_strides(rank), output_strides(rank);
    std::vector<bool> reduction_axes(rank);
    for (auto& extent : source_shape) extent = read();
    for (auto& stride : source_strides) stride = read_stride();
    for (auto& extent : output_shape) extent = read();
    for (auto& stride : output_strides) stride = read_stride();
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto flag = read();
      if (flag > 1) {
        throw std::invalid_argument("reduction layout axis flag is not boolean");
      }
      reduction_axes[axis] = flag != 0;
    }
    auto record = layout::BuildReductionLayout(
        source_shape, source_strides, source_offset, output_shape, output_strides,
        output_offset, reduction_axes, decoded.source_size, decoded.output_size, index);
    layout::OptimizeRecordForExecution(&record);
    decoded.records.push_back(std::move(record));
  }
  if (cursor != byte_count) {
    throw std::invalid_argument("reduction layout descriptor has trailing words");
  }
  return decoded;
}

}

std::atomic<uint64_t> prepared_created_count{0};
std::atomic<uint64_t> prepared_destroyed_count{0};

PreparedState::PreparedState(descriptor::DecodedLayout value)
    : records(std::move(value.records)), source_size(value.source_size), output_size(value.output_size) {
  prepared_created_count.fetch_add(1, std::memory_order_relaxed);
}

PreparedState::~PreparedState() {
  prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
}

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiatePrepared(ffi::Span<const int64_t> words) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    state = std::make_unique<PreparedState>(descriptor::DecodeLayout(words.begin(), words.size()));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateReduction(
    ffi::Span<const uint8_t> bytes, ffi::Span<const int64_t>) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    state = std::make_unique<PreparedState>(descriptor::DecodeReductionLayout(
        reinterpret_cast<const char*>(bytes.begin()), bytes.size()));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateDot(
    ffi::Span<const int64_t> words, int64_t) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto layout = descriptor::DecodeAddressLayout(words.begin(), words.size());
    for (auto& record : layout.records) layout::OptimizeRecordForExecution(&record);
    state = std::make_unique<PreparedState>(std::move(layout));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

ffi::ErrorOr<std::unique_ptr<PreparedState>> InstantiateAccumulation(
    ffi::Span<const int64_t> words, ffi::Span<const int64_t>) {
  std::unique_ptr<PreparedState> state;
  const auto error = ContainErrors([&] {
    auto layout = descriptor::DecodeAddressLayout(words.begin(), words.size());
    for (auto& record : layout.records) layout::OptimizeRecordForExecution(&record);
    state = std::make_unique<PreparedState>(std::move(layout));
  });
  if (error.failure()) return ffi::Unexpected(error);
  return state;
}

uint64_t BufferBytes(uint64_t elements, uint64_t item_size) {
  if (elements > static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / item_size) {
    throw std::invalid_argument("buffer size exceeds addressable storage");
  }
  return elements * item_size;
}

void ValidateDisjointBuffers(const void* input, uint64_t input_bytes,
                              const void* output, uint64_t output_bytes) {
  if (input_bytes == 0 || output_bytes == 0) return;
  const auto input_start = reinterpret_cast<std::uintptr_t>(input);
  const auto output_start = reinterpret_cast<std::uintptr_t>(output);
  const bool overlap = input_start <= output_start
      ? output_start - input_start < input_bytes
      : input_start - output_start < output_bytes;
  if (overlap) throw std::invalid_argument("input and output buffers overlap");
}

void ValidatePreparedDimensions(const PreparedState& prepared, uint64_t source_size, uint64_t output_size) {
  if (prepared.source_size != source_size || prepared.output_size != output_size) {
    throw std::invalid_argument("buffer dimensions do not match layout");
  }
}

}

#ifndef TENSOR0_STRIDE_JAX_VERSION
#define TENSOR0_STRIDE_JAX_VERSION "unknown"
#endif

#ifndef TENSOR0_STRIDE_JAXLIB_VERSION
#define TENSOR0_STRIDE_JAXLIB_VERSION "unknown"
#endif

namespace ffi = xla::ffi;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideDotInstantiateV1, tensor0::stride::InstantiateDot,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<int64_t>("conjugate_left"));

extern "C" void* Tensor0StrideDotInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideDotInstantiateV1);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideInstantiateV1, tensor0::stride::InstantiatePrepared,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout"));

extern "C" void* Tensor0StrideInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideInstantiateV1);
}

extern "C" void* Tensor0StridePreparedTypeId() {
  return reinterpret_cast<void*>(&tensor0::stride::PreparedState::id);
}

extern "C" const void* Tensor0StridePreparedTypeInfo() {
  return reinterpret_cast<const void*>(&tensor0::stride::kPreparedTypeInfo);
}

extern "C" uint64_t Tensor0StridePreparedCreatedCount() {
  return tensor0::stride::prepared_created_count.load(std::memory_order_relaxed);
}

extern "C" uint64_t Tensor0StridePreparedDestroyedCount() {
  return tensor0::stride::prepared_destroyed_count.load(std::memory_order_relaxed);
}

extern "C" const char* Tensor0StrideBuiltJaxVersion() {
  return TENSOR0_STRIDE_JAX_VERSION;
}

extern "C" const char* Tensor0StrideBuiltJaxlibVersion() {
  return TENSOR0_STRIDE_JAXLIB_VERSION;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideReductionInstantiateV1, tensor0::stride::InstantiateReduction,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const uint8_t>>("layout")
        .Attr<ffi::Span<const int64_t>>("coefficient_records"));

extern "C" void* Tensor0StrideReductionInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideReductionInstantiateV1);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Tensor0StrideAccumulationInstantiateV1, tensor0::stride::InstantiateAccumulation,
    ffi::Ffi::BindInstantiate().Attr<ffi::Span<const int64_t>>("layout")
        .Attr<ffi::Span<const int64_t>>("coefficient_records"));

extern "C" void* Tensor0StrideAccumulationInstantiateV1Handler() {
  return reinterpret_cast<void*>(&Tensor0StrideAccumulationInstantiateV1);
}
