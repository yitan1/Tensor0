namespace descriptor {

inline constexpr int64_t kLayoutVersion = 1;

struct DecodedLayout {
  std::vector<layout::Record> records;
  uint64_t source_size = 0;
  uint64_t output_size = 0;
};

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

inline constexpr uint64_t kReductionLayoutVersion = 1;

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
