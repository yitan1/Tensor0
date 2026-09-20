#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor0::stride::layout {

struct Record {
  std::size_t semantic_index = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  std::vector<uint64_t> shape;
  std::vector<int64_t> source_strides;
  std::vector<int64_t> destination_strides;
};

bool CheckedAdd(uint64_t left, uint64_t right, uint64_t* result);

bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t* result);

uint64_t ElementCount(const Record& record);

uint64_t AbsoluteStride(int64_t stride);

bool StrideEquals(int64_t stride, uint64_t value);

bool PositiveStrideValue(int64_t stride, uint64_t* value);

bool CheckedMultiplyStride(int64_t stride, uint64_t extent,
                           int64_t* result);

const std::vector<int64_t>& Strides(const Record& record, bool source);

int64_t Offset(const Record& record, bool source);

void AddressBounds(const Record& record, bool source,
                   const std::vector<std::size_t>& axes,
                   int64_t* minimum, int64_t* maximum);

void ValidateInjectiveView(const Record& record, bool source,
                           std::vector<std::size_t> axes);

Record BuildLayout(
    const std::vector<uint64_t>& shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<int64_t>& destination_strides, int64_t destination_offset,
    uint64_t source_size, uint64_t output_size, std::size_t semantic_index);

Record BuildReductionLayout(
    const std::vector<uint64_t>& source_shape,
    const std::vector<int64_t>& source_strides, int64_t source_offset,
    const std::vector<uint64_t>& output_shape,
    const std::vector<int64_t>& output_strides, int64_t output_offset,
    const std::vector<bool>& reduction_axes, uint64_t source_size,
    uint64_t output_size, std::size_t semantic_index);

Record SliceRecordRange(const Record& record, std::size_t axis,
                            uint64_t unit_start, uint64_t unit_count);

}
