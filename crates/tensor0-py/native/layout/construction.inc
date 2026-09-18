namespace layout {

Record BuildLayout(
    const std::vector<uint64_t>& shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<int64_t>& destination_strides, int64_t destination_offset,
    uint64_t source_size, uint64_t output_size, std::size_t semantic_index) {
  if (source_strides.size() != shape.size() || destination_strides.size() != shape.size()) {
    throw std::invalid_argument("layout field lengths mismatch");
  }
  Record record{semantic_index, source_offset, destination_offset, shape,
                source_strides, destination_strides};
  const auto element_count = ElementCount(record);
  std::vector<std::size_t> axes(element_count == 0 ? 0 : record.shape.size());
  std::iota(axes.begin(), axes.end(), 0);
  for (bool source : {true, false}) {
    int64_t minimum = 0;
    int64_t maximum = 0;
    AddressBounds(record, source, axes, &minimum, &maximum);
    const uint64_t size = source ? source_size : output_size;
    if ((element_count == 0 && static_cast<uint64_t>(maximum) > size) ||
        (element_count != 0 && static_cast<uint64_t>(maximum) >= size)) {
      throw std::invalid_argument(std::string(source ? "source" : "destination") +
                                  " address exceeds storage");
    }
  }
  return record;
}

Record BuildReductionLayout(
    const std::vector<uint64_t>& source_shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<uint64_t>& output_shape,
    const std::vector<int64_t>& output_strides, int64_t output_offset,
    const std::vector<bool>& reduction_axes, uint64_t source_size,
    uint64_t output_size, std::size_t semantic_index) {
  if (source_strides.size() != source_shape.size() ||
      output_shape.size() != source_shape.size() ||
      output_strides.size() != source_shape.size() ||
      reduction_axes.size() != source_shape.size()) {
    throw std::invalid_argument("reduction layout field lengths mismatch");
  }
  Record record{semantic_index, source_offset, output_offset, source_shape,
                source_strides, output_strides};
  if (source_offset < 0 || output_offset < 0) {
    throw std::invalid_argument("reduction address precedes storage");
  }
  uint64_t output_count = 1;
  uint64_t reduction_count = 1;
  std::vector<std::size_t> map_axes;
  for (std::size_t axis = 0; axis < record.shape.size(); ++axis) {
    const uint64_t extent = record.shape[axis];
    const bool reduction = reduction_axes[axis];
    if (output_shape[axis] != (reduction ? 1 : extent)) {
      throw std::invalid_argument("reduction input/output shapes do not match axis roles");
    }
    if (reduction) record.destination_strides[axis] = 0;
    if (!reduction && extent > 1 && record.destination_strides[axis] == 0) {
      throw std::invalid_argument("reduction output map has a zero nontrivial stride");
    }
    uint64_t* count = reduction ? &reduction_count : &output_count;
    if (!CheckedMultiply(*count, extent, count)) {
      throw std::invalid_argument("reduction fiber element count overflow");
    }
    if (!reduction) map_axes.push_back(axis);
  }
  uint64_t logical_count = 0;
  if (!CheckedMultiply(output_count, reduction_count, &logical_count)) {
    throw std::invalid_argument("reduction logical element count overflow");
  }
  ValidateInjectiveView(record, false, map_axes);
  std::vector<std::size_t> source_axes(record.shape.size());
  std::iota(source_axes.begin(), source_axes.end(), 0);
  for (bool source : {true, false}) {
    const bool empty = source ? logical_count == 0 : output_count == 0;
    const uint64_t size = source ? source_size : output_size;
    if (empty) {
      if (static_cast<uint64_t>(Offset(record, source)) > size) {
        throw std::invalid_argument("empty reduction offset exceeds storage");
      }
    } else {
      int64_t minimum = 0;
      int64_t maximum = 0;
      AddressBounds(record, source, source ? source_axes : map_axes,
                    &minimum, &maximum);
      if (static_cast<uint64_t>(maximum) >= size) {
        throw std::invalid_argument("reduction address exceeds storage");
      }
    }
  }
  return record;
}

}
