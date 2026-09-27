#include "descriptor.h"
#include "owner_fiber.h"

#include <algorithm>
#include <cstring>
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

DecodedLayout DecodeAccumulationLayout(const int64_t* words, std::size_t word_count) {
  auto decoded = DecodeAddressLayout(words, word_count);
  for (const auto& record : decoded.records) {
    if (layout::ElementCount(record) == 0) continue;
    std::vector<std::size_t> axes;
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      if (record.shape[axis] > 1 && record.destination_strides[axis] != 0) axes.push_back(axis);
    }
    try {
      layout::ValidateInjectiveView(record, false, std::move(axes));
    } catch (const std::invalid_argument&) {
      throw std::invalid_argument("accumulation cannot prove injective output owners");
    }
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
  }
  return decoded;
}

DecodedReductionLayout DecodeReductionLayout(const char* bytes, std::size_t byte_count) {
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
  DecodedReductionLayout decoded;
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
    // Preserve original-axis prefix overflow validation previously performed by
    // CPU optimization, even when grouped map/fiber products become zero.
    layout::ElementCount(record);
    decoded.records.push_back({std::move(record), std::move(reduction_axes)});
  }
  if (cursor != byte_count) {
    throw std::invalid_argument("reduction layout descriptor has trailing words");
  }
  return decoded;
}

DecodedLayout PrepareLayout(const int64_t* words, std::size_t word_count) {
  auto decoded = DecodeLayout(words, word_count);
  for (auto& record : decoded.records) layout::OptimizeRecordForPreparation(record);
  return decoded;
}

DecodedLayout PrepareAddressLayout(const int64_t* words, std::size_t word_count) {
  auto decoded = DecodeAddressLayout(words, word_count);
  for (auto& record : decoded.records) layout::OptimizeRecordForPreparation(record);
  return decoded;
}

DecodedLayout PrepareAccumulationLayout(const int64_t* words, std::size_t word_count) {
  auto decoded = DecodeAccumulationLayout(words, word_count);
  for (auto& record : decoded.records) layout::OptimizeRecordForPreparation(record);
  return decoded;
}

DecodedReductionLayout PrepareReductionLayout(const char* bytes, std::size_t byte_count) {
  auto decoded = DecodeReductionLayout(bytes, byte_count);
  for (auto& record : decoded.records) layout::OptimizeRecordForPreparation(record);
  return decoded;
}

std::vector<int64_t> EncodeLayout(const DecodedLayout& decoded) {
  std::vector<int64_t> words{
      kLayoutVersion, static_cast<int64_t>(decoded.source_size),
      static_cast<int64_t>(decoded.output_size), static_cast<int64_t>(decoded.records.size())};
  for (const auto& record : decoded.records) {
    words.insert(words.end(), {static_cast<int64_t>(record.shape.size()),
                              record.source_offset, record.destination_offset});
    for (auto extent : record.shape) words.push_back(static_cast<int64_t>(extent));
    words.insert(words.end(), record.source_strides.begin(), record.source_strides.end());
    words.insert(words.end(), record.destination_strides.begin(), record.destination_strides.end());
  }
  return words;
}

std::vector<char> EncodeReductionLayout(const DecodedReductionLayout& decoded) {
  std::vector<char> bytes;
  const auto append = [&](uint64_t word) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      const auto byte = static_cast<unsigned char>((word >> shift) & 255);
      bytes.push_back(static_cast<char>(byte));
    }
  };
  append(kReductionLayoutVersion);
  append(decoded.source_size);
  append(decoded.output_size);
  append(decoded.records.size());
  for (const auto& reduction : decoded.records) {
    const auto& record = reduction.layout;
    append(record.shape.size());
    append(static_cast<uint64_t>(record.source_offset));
    append(static_cast<uint64_t>(record.destination_offset));
    for (auto extent : record.shape) append(extent);
    for (auto stride : record.source_strides) append(static_cast<uint64_t>(stride));
    for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
      append(reduction.reduction_axes[axis] ? 1 : record.shape[axis]);
    }
    for (auto stride : record.destination_strides) append(static_cast<uint64_t>(stride));
    for (bool reduction_axis : reduction.reduction_axes) append(reduction_axis);
  }
  return bytes;
}

}

namespace layout {
namespace {
OwnerFiberLayout Pack(const std::vector<Record>& records,
                      const std::vector<std::vector<bool>>& roles,
                      uint64_t source_size, uint64_t output_size, int32_t operation) {
  OwnerFiberLayout packed{{1, static_cast<int64_t>(source_size),
                           static_cast<int64_t>(output_size), static_cast<int64_t>(records.size())}, {}};
  packed.schedules.reserve(records.size());
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto& record = records[i];
    const auto rank = record.shape.size();
    std::vector<std::size_t> map, fiber;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const bool is_fiber = operation == 4 ? roles[i][axis]
          : operation == 2 || (operation == 3 &&
              record.destination_strides[axis] == 0 && record.shape[axis] > 1);
      (is_fiber ? fiber : map).push_back(axis);
    }
    const bool empty = ElementCount(record) == 0;
    uint64_t owners = empty && operation != 4 && operation != 2 ? 0 : 1;
    uint64_t contributions = empty && operation != 4 ? 0 : 1;
    if (!empty || operation == 4) {
      for (const auto axis : map) owners *= record.shape[axis];
      for (const auto axis : fiber) contributions *= record.shape[axis];
    }
    packed.schedules.push_back({owners, contributions, packed.words.size()});
    packed.words.insert(packed.words.end(), {static_cast<int64_t>(map.size()),
        static_cast<int64_t>(fiber.size()), record.source_offset, record.destination_offset});
    const auto append_axes = [&](const auto& values) {
      for (const auto axis : map) packed.words.push_back(static_cast<int64_t>(values[axis]));
      for (const auto axis : fiber) packed.words.push_back(static_cast<int64_t>(values[axis]));
    };
    append_axes(record.shape);
    append_axes(record.source_strides);
    append_axes(record.destination_strides);
  }
  return packed;
}
}  // namespace

OwnerFiberLayout PackOwnerFiber(const descriptor::DecodedLayout& decoded, int32_t operation) {
  return Pack(decoded.records, {}, decoded.source_size, decoded.output_size, operation);
}

OwnerFiberLayout PackOwnerFiber(const descriptor::DecodedReductionLayout& decoded) {
  std::vector<Record> records;
  std::vector<std::vector<bool>> roles;
  records.reserve(decoded.records.size());
  roles.reserve(decoded.records.size());
  for (const auto& record : decoded.records) {
    records.push_back(record.layout);
    roles.push_back(record.reduction_axes);
  }
  return Pack(records, roles, decoded.source_size, decoded.output_size, 4);
}
}  // namespace layout

}  // namespace tensor0::stride

extern "C" void Tensor0StridePrepareLayout(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept {
  namespace descriptor = tensor0::stride::descriptor;
  try {
    std::vector<int64_t> output;
    switch (operation) {
      case 0:
      case 1: {
        output = descriptor::EncodeLayout(descriptor::PrepareLayout(words, word_count));
        break;
      }
      case 2: {
        output = descriptor::EncodeLayout(descriptor::PrepareAddressLayout(words, word_count));
        break;
      }
      case 3: {
        output = descriptor::EncodeLayout(descriptor::PrepareAccumulationLayout(words, word_count));
        break;
      }
      case 4: {
        std::vector<char> bytes;
        if (word_count > bytes.max_size() / sizeof(int64_t)) {
          throw std::invalid_argument("reduction layout descriptor is too large");
        }
        bytes.reserve(word_count * sizeof(int64_t));
        for (std::size_t i = 0; i < word_count; ++i) {
          const auto word = static_cast<uint64_t>(words[i]);
          for (unsigned shift = 0; shift < 64; shift += 8) {
            bytes.push_back(static_cast<char>((word >> shift) & 255));
          }
        }
        bytes = descriptor::EncodeReductionLayout(
            descriptor::PrepareReductionLayout(bytes.data(), bytes.size()));
        output.reserve(bytes.size() / sizeof(int64_t));
        for (std::size_t i = 0; i < bytes.size(); i += sizeof(int64_t)) {
          uint64_t word = 0;
          for (unsigned byte = 0; byte < 8; ++byte) {
            word |= static_cast<uint64_t>(static_cast<uint8_t>(bytes[i + byte])) << (8 * byte);
          }
          output.push_back(word <= static_cast<uint64_t>(INT64_MAX)
              ? static_cast<int64_t>(word) : -1 - static_cast<int64_t>(UINT64_MAX - word));
        }
        break;
      }
      default:
        throw std::invalid_argument("unsupported stride layout operation");
    }
    callback(context, 0, output.data(), output.size(), nullptr, 0);
  } catch (const std::invalid_argument& error) {
    callback(context, 1, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (const std::exception& error) {
    callback(context, 2, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (...) {
    constexpr char error[] = "native stride layout preparation failed";
    callback(context, 2, nullptr, 0, error, sizeof(error) - 1);
  }
}

extern "C" void Tensor0StrideSumScratchCapacity(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept {
  try {
    namespace descriptor = tensor0::stride::descriptor;
    namespace layout = tensor0::stride::layout;
    layout::OwnerFiberLayout packed;
    if (operation == 3) {
      packed = layout::PackOwnerFiber(descriptor::DecodeAccumulationLayout(words, word_count));
    } else if (operation == 4) {
      std::vector<char> bytes;
      if (word_count > bytes.max_size() / sizeof(int64_t))
        throw std::invalid_argument("reduction layout descriptor is too large");
      bytes.reserve(word_count * sizeof(int64_t));
      for (std::size_t i = 0; i < word_count; ++i) {
        const auto word = static_cast<uint64_t>(words[i]);
        for (unsigned shift = 0; shift < 64; shift += 8)
          bytes.push_back(static_cast<char>((word >> shift) & 255));
      }
      packed = layout::PackOwnerFiber(descriptor::DecodeReductionLayout(bytes.data(), bytes.size()));
    } else {
      throw std::invalid_argument("unsupported sum scratch operation");
    }
    const int64_t capacity = static_cast<int64_t>(layout::SumScratchCapacity(packed.schedules));
    callback(context, 0, &capacity, 1, nullptr, 0);
  } catch (const std::invalid_argument& error) {
    callback(context, 1, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (const std::exception& error) {
    callback(context, 2, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (...) {
    constexpr char error[] = "native sum scratch preparation failed";
    callback(context, 2, nullptr, 0, error, sizeof(error) - 1);
  }
}

extern "C" void Tensor0StrideDotScratchCapacity(
    const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept {
  try {
    const auto decoded = tensor0::stride::descriptor::DecodeAddressLayout(words, word_count);
    const auto packed = tensor0::stride::layout::PackOwnerFiber(decoded, 2);
    const int64_t capacity = static_cast<int64_t>(
        tensor0::stride::layout::DotScratchCapacity(packed.schedules));
    callback(context, 0, &capacity, 1, nullptr, 0);
  } catch (const std::invalid_argument& error) {
    callback(context, 1, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (const std::exception& error) {
    callback(context, 2, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (...) {
    constexpr char error[] = "native dot scratch preparation failed";
    callback(context, 2, nullptr, 0, error, sizeof(error) - 1);
  }
}

extern "C" void Tensor0StridePackOwnerFiber(
    int32_t operation, const int64_t* words, std::size_t word_count,
    void* context, Tensor0StridePrepareLayoutCallback callback) noexcept {
  namespace descriptor = tensor0::stride::descriptor;
  namespace layout = tensor0::stride::layout;
  try {
    layout::OwnerFiberLayout packed;
    if (operation == 0 || operation == 1) {
      packed = layout::PackOwnerFiber(descriptor::DecodeLayout(words, word_count), operation);
    } else if (operation == 2) {
      packed = layout::PackOwnerFiber(descriptor::DecodeAddressLayout(words, word_count), operation);
    } else if (operation == 3) {
      packed = layout::PackOwnerFiber(descriptor::DecodeAccumulationLayout(words, word_count));
    } else if (operation == 4) {
      std::vector<char> bytes;
      if (word_count > bytes.max_size() / sizeof(int64_t))
        throw std::invalid_argument("reduction layout descriptor is too large");
      bytes.reserve(word_count * sizeof(int64_t));
      for (std::size_t i = 0; i < word_count; ++i) {
        const auto word = static_cast<uint64_t>(words[i]);
        for (unsigned shift = 0; shift < 64; shift += 8)
          bytes.push_back(static_cast<char>((word >> shift) & 255));
      }
      packed = layout::PackOwnerFiber(descriptor::DecodeReductionLayout(bytes.data(), bytes.size()));
    } else {
      throw std::invalid_argument("unsupported owner/fiber operation");
    }
    callback(context, 0, packed.words.data(), packed.words.size(), nullptr, 0);
  } catch (const std::invalid_argument& error) {
    callback(context, 1, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (const std::exception& error) {
    callback(context, 2, nullptr, 0, error.what(), std::strlen(error.what()));
  } catch (...) {
    constexpr char error[] = "native owner/fiber encoding failed";
    callback(context, 2, nullptr, 0, error, sizeof(error) - 1);
  }
}
