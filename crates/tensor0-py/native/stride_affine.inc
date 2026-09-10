uint64_t GenericRowCount(const Record& record) {
  return record.logical_elements /
         record.shape[record.loop_axes_fastest_first[0]];
}

struct AssignMappedValue {
  template <typename DestinationElement, typename MappedElement>
  void operator()(DestinationElement& destination,
                  MappedElement mapped) const {
    destination = static_cast<DestinationElement>(mapped);
  }
};

template <typename WriteOp>
inline constexpr bool kIsAssignMappedValue =
    std::is_same_v<std::decay_t<WriteOp>, AssignMappedValue>;

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteRank2AffineColumns(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t first_column, uint64_t column_count, ScalarOp scalar_op,
    WriteOp write_op) {
  constexpr uint64_t tile = 32;
  const uint64_t rows = record.shape[0];
  const uint64_t column_stop_limit = first_column + column_count;
  for (uint64_t column_start = first_column;
       column_start < column_stop_limit; column_start += tile) {
    const uint64_t column_stop =
        std::min(column_stop_limit, column_start + tile);
    for (uint64_t row_start = 0; row_start < rows; row_start += tile) {
      const uint64_t row_stop = std::min(rows, row_start + tile);
      for (uint64_t column = column_start; column < column_stop; ++column) {
        int64_t source_index =
            record.source_offset +
            static_cast<int64_t>(row_start) * record.source_strides[0] +
            static_cast<int64_t>(column) * record.source_strides[1];
        int64_t destination_index =
            record.destination_offset +
            static_cast<int64_t>(row_start) *
                record.destination_strides[0] +
            static_cast<int64_t>(column) * record.destination_strides[1];
        for (uint64_t row = row_start; row < row_stop; ++row) {
          write_op(result[destination_index],
                   scalar_op(source[source_index]));
          source_index += record.source_strides[0];
          destination_index += record.destination_strides[0];
        }
      }
    }
  }
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteRank2AffineRows(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t first_row, uint64_t row_count, ScalarOp scalar_op,
    WriteOp write_op) {
  constexpr uint64_t tile = 32;
  const uint64_t columns = record.shape[1];
  const uint64_t row_stop_limit = first_row + row_count;
  for (uint64_t row_start = first_row; row_start < row_stop_limit;
       row_start += tile) {
    const uint64_t row_stop = std::min(row_stop_limit, row_start + tile);
    for (uint64_t column_start = 0; column_start < columns;
         column_start += tile) {
      const uint64_t column_stop =
          std::min(columns, column_start + tile);
      for (uint64_t row = row_start; row < row_stop; ++row) {
        int64_t source_index =
            record.source_offset +
            static_cast<int64_t>(row) * record.source_strides[0] +
            static_cast<int64_t>(column_start) * record.source_strides[1];
        int64_t destination_index =
            record.destination_offset +
            static_cast<int64_t>(row) * record.destination_strides[0] +
            static_cast<int64_t>(column_start) *
                record.destination_strides[1];
        for (uint64_t column = column_start; column < column_stop; ++column) {
          write_op(result[destination_index],
                   scalar_op(source[source_index]));
          source_index += record.source_strides[1];
          destination_index += record.destination_strides[1];
        }
      }
    }
  }
}

template <typename RowOp>
void ExecuteGenericAffineRows(const Record& record, uint64_t row_start,
                              uint64_t row_count, RowOp row_op) {
  if (row_count == 0) {
    return;
  }
  const std::size_t fastest_axis = record.loop_axes_fastest_first[0];
  const uint64_t inner_elements = record.shape[fastest_axis];
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
    row_op(source_base, destination_base, inner_elements);
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

template <typename RowOp>
void ExecuteGenericAffineRowSegmentWithOp(
    const Record& record, uint64_t row_index, uint64_t inner_start,
    uint64_t inner_count, RowOp row_op) {
  const std::size_t fastest_axis =
      record.loop_axes_fastest_first[0];
  int64_t source_base =
      record.source_offset +
      static_cast<int64_t>(inner_start) *
          record.source_strides[fastest_axis];
  int64_t destination_base =
      record.destination_offset +
      static_cast<int64_t>(inner_start) *
          record.destination_strides[fastest_axis];
  uint64_t remaining = row_index;
  for (std::size_t position = 1; position < record.rank; ++position) {
    const std::size_t axis =
        record.loop_axes_fastest_first[position];
    const uint64_t coordinate = remaining % record.shape[axis];
    remaining /= record.shape[axis];
    source_base += static_cast<int64_t>(coordinate) *
                   record.source_strides[axis];
    destination_base += static_cast<int64_t>(coordinate) *
                        record.destination_strides[axis];
  }
  row_op(source_base, destination_base, inner_count);
}

template <typename RowOp>
void ExecuteGenericAffineElementRangeWithOp(
    const Record& record, uint64_t element_start,
    uint64_t element_count, RowOp row_op) {
  if (element_count == 0) {
    return;
  }
  const std::size_t fastest_axis =
      record.loop_axes_fastest_first[0];
  const uint64_t inner_elements = record.shape[fastest_axis];
  uint64_t row_start = element_start / inner_elements;
  const uint64_t inner_start = element_start % inner_elements;
  uint64_t remaining = element_count;
  if (inner_start != 0) {
    const uint64_t segment_count =
        std::min(remaining, inner_elements - inner_start);
    ExecuteGenericAffineRowSegmentWithOp(
        record, row_start, inner_start, segment_count, row_op);
    remaining -= segment_count;
    ++row_start;
  }
  const uint64_t full_rows = remaining / inner_elements;
  if (full_rows != 0) {
    ExecuteGenericAffineRows(record, row_start, full_rows, row_op);
    remaining -= full_rows * inner_elements;
    row_start += full_rows;
  }
  if (remaining != 0) {
    ExecuteGenericAffineRowSegmentWithOp(
        record, row_start, 0, remaining, row_op);
  }
}

template <bool kUnitScale, typename SourceElement, typename ResultElement,
          typename ScalarOp, typename WriteOp>
void ExecuteGenericAffineScalarRow(
    const Record& record, const SourceElement* source, ResultElement* result,
    int64_t source_base, int64_t destination_base, uint64_t inner_count,
    int64_t source_inner_stride, int64_t destination_inner_stride,
    ScalarOp scalar_op, WriteOp write_op) {
  int64_t source_index = source_base;
  int64_t destination_index = destination_base;
  for (uint64_t inner = 0; inner < inner_count; ++inner) {
    if constexpr (kUnitScale) {
      write_op(result[destination_index], source[source_index]);
    } else {
      write_op(result[destination_index],
               scalar_op(source[source_index]));
    }
    source_index += source_inner_stride;
    destination_index += destination_inner_stride;
  }
}

template <bool kUnitScale, typename SourceElement, typename ResultElement,
          typename ScalarOp, typename WriteOp>
void ExecuteGenericAffineRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t row_start, uint64_t row_count, ScalarOp scalar_op,
    WriteOp write_op) {
  const std::size_t fastest_axis =
      record.loop_axes_fastest_first[0];
  const int64_t source_inner_stride =
      record.source_strides[fastest_axis];
  const int64_t destination_inner_stride =
      record.destination_strides[fastest_axis];
  const auto row_op = [&](int64_t source_base, int64_t destination_base,
                          uint64_t inner_count) {
    ExecuteGenericAffineScalarRow<kUnitScale>(
        record, source, result, source_base, destination_base,
        inner_count, source_inner_stride, destination_inner_stride,
        scalar_op, write_op);
  };
  ExecuteGenericAffineRows(record, row_start, row_count, row_op);
}

template <bool kUnitScale, typename SourceElement, typename ResultElement,
          typename ScalarOp>
void ExecuteBroadcastContiguousRows(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t row_start, uint64_t row_count, ScalarOp scalar_op) {
  const auto row_op = [&](int64_t source_base, int64_t destination_base,
                          uint64_t inner_count) {
    const auto mapped = [&] {
      if constexpr (kUnitScale) {
        return source[source_base];
      } else {
        return scalar_op(source[source_base]);
      }
    }();
    ResultElement* output = result + destination_base;
    std::fill(output, output + inner_count,
              static_cast<ResultElement>(mapped));
  };
  ExecuteGenericAffineRows(record, row_start, row_count, row_op);
}

template <bool kUnitScale, typename SourceElement, typename ResultElement,
          typename ScalarOp, typename WriteOp>
void ExecuteGenericAffineElementRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    uint64_t element_start, uint64_t element_count, ScalarOp scalar_op,
    WriteOp write_op) {
  const std::size_t fastest_axis =
      record.loop_axes_fastest_first[0];
  const int64_t source_inner_stride =
      record.source_strides[fastest_axis];
  const int64_t destination_inner_stride =
      record.destination_strides[fastest_axis];
  const auto row_op = [&](int64_t source_base, int64_t destination_base,
                          uint64_t inner_count) {
    ExecuteGenericAffineScalarRow<kUnitScale>(
        record, source, result, source_base, destination_base,
        inner_count, source_inner_stride, destination_inner_stride,
        scalar_op, write_op);
  };
  ExecuteGenericAffineElementRangeWithOp(
      record, element_start, element_count, row_op);
}

template <typename SourceElement, typename ResultElement, typename MapOp,
          typename WriteOp>
TENSOR0_STRIDE_ALWAYS_INLINE void ExecuteContiguousMap(
    const SourceElement* input, ResultElement* output,
    uint64_t element_count, MapOp map_op, WriteOp write_op) {
  for (uint64_t element = 0; element < element_count; ++element) {
    write_op(output[element], map_op(input[element]));
  }
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteAffineLeafRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    LeafKernel kind, uint64_t unit_start, uint64_t unit_count,
    ScalarOp scalar_op, WriteOp write_op);



#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_F16C_TARGET
bool ExecuteContiguousF16F32Avx2(const uint16_t* input, float* output,
                                 uint64_t element_count, ScalarOp scalar_op) {
  constexpr bool unit_scale = std::is_same_v<ScalarOp, ScalarCast<ffi::F16, ffi::F32>>;
  const float scale = [&] {
    if constexpr (unit_scale) return 1.0F;
    else return scalar_op.factor;
  }();
  const __m256 scale8 = _mm256_set1_ps(scale);
  uint64_t element = 0;
  for (; element + 8 <= element_count; element += 8) {
    const __m128i half =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + element));
    __m256 values = _mm256_cvtph_ps(half);
    if constexpr (!unit_scale) {
      values = _mm256_mul_ps(values, scale8);
    }
    _mm256_storeu_ps(output + element, values);
  }
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], scalar_op(input[element]));
  }
  return true;
}

#endif

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_F16C_TARGET
bool ExecuteContiguousF16F16Avx2(const uint16_t* input, uint16_t* output,
                                 uint64_t element_count, ScalarOp scalar_op) {
  if constexpr (std::is_same_v<ScalarOp, IdentityScalar>) {
    std::memcpy(output, input, element_count * sizeof(uint16_t));
    return true;
  } else {
    const float scale = HalfToFloat(scalar_op.factor);
    const __m256 scale8 = _mm256_set1_ps(scale);
    uint64_t element = 0;
    for (; element + 8 <= element_count; element += 8) {
      const __m128i half =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + element));
      const __m256 values = _mm256_mul_ps(_mm256_cvtph_ps(half), scale8);
      const __m128i result = _mm256_cvtps_ph(
          values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(output + element), result);
    }
    for (; element < element_count; ++element) {
      AssignMappedValue{}(
          output[element], scalar_op(input[element]));
    }
    return true;
  }
}

TENSOR0_STRIDE_AVX2_F16C_TARGET
void ExecuteContiguousF32F16TransposeAvx2(
    const float* input, uint16_t* output,
    uint64_t element_count, NarrowF16Product scalar_op) {
  const __m256 scale = _mm256_set1_ps(
      scalar_op.factor);
  uint64_t element = 0;
  for (; element + 8 <= element_count; element += 8) {
    const __m256 mapped =
        _mm256_mul_ps(_mm256_loadu_ps(input + element), scale);
    const __m128i half =
        _mm256_cvtps_ph(
            mapped, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + element), half);
  }
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], scalar_op(input[element]));
  }
}

template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_F16C_TARGET
__m128i ScaleF16x8(const uint16_t* input, ScalarOp scalar_op) {
  const __m128i half =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
  if constexpr (std::is_same_v<ScalarOp, IdentityScalar>) {
    return half;
  }
  else {
  const __m256 scale = _mm256_set1_ps(HalfToFloat(scalar_op.factor));
  return _mm256_cvtps_ph(
      _mm256_mul_ps(_mm256_cvtph_ps(half), scale),
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
}

TENSOR0_STRIDE_AVX2_F16C_TARGET
void Transpose8x8F16(__m128i* values) {
  const __m128i t0 = _mm_unpacklo_epi16(values[0], values[1]);
  const __m128i t1 = _mm_unpackhi_epi16(values[0], values[1]);
  const __m128i t2 = _mm_unpacklo_epi16(values[2], values[3]);
  const __m128i t3 = _mm_unpackhi_epi16(values[2], values[3]);
  const __m128i t4 = _mm_unpacklo_epi16(values[4], values[5]);
  const __m128i t5 = _mm_unpackhi_epi16(values[4], values[5]);
  const __m128i t6 = _mm_unpacklo_epi16(values[6], values[7]);
  const __m128i t7 = _mm_unpackhi_epi16(values[6], values[7]);
  const __m128i u0 = _mm_unpacklo_epi32(t0, t2);
  const __m128i u1 = _mm_unpackhi_epi32(t0, t2);
  const __m128i u2 = _mm_unpacklo_epi32(t1, t3);
  const __m128i u3 = _mm_unpackhi_epi32(t1, t3);
  const __m128i u4 = _mm_unpacklo_epi32(t4, t6);
  const __m128i u5 = _mm_unpackhi_epi32(t4, t6);
  const __m128i u6 = _mm_unpacklo_epi32(t5, t7);
  const __m128i u7 = _mm_unpackhi_epi32(t5, t7);
  values[0] = _mm_unpacklo_epi64(u0, u4);
  values[1] = _mm_unpackhi_epi64(u0, u4);
  values[2] = _mm_unpacklo_epi64(u1, u5);
  values[3] = _mm_unpackhi_epi64(u1, u5);
  values[4] = _mm_unpacklo_epi64(u2, u6);
  values[5] = _mm_unpackhi_epi64(u2, u6);
  values[6] = _mm_unpacklo_epi64(u3, u7);
  values[7] = _mm_unpackhi_epi64(u3, u7);
}

template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_F16C_TARGET
void ExecuteRank2F16ForwardAvx2(const Record& record,
                                const uint16_t* source, uint16_t* result,
                                uint64_t first_column,
                                uint64_t column_count, ScalarOp scalar_op) {
  constexpr uint64_t tile = 8;
  const uint64_t rows = record.shape[0];
  const uint64_t column_stop = first_column + column_count;
  for (uint64_t column_start = first_column; column_start < column_stop;
       column_start += tile) {
    const uint64_t columns_in_tile =
        std::min(tile, column_stop - column_start);
    for (uint64_t row_start = 0; row_start < rows; row_start += tile) {
      const uint64_t rows_in_tile = std::min(tile, rows - row_start);
      if (rows_in_tile == tile && columns_in_tile == tile) {
        __m128i values[tile];
        for (uint64_t column = 0; column < tile; ++column) {
          const int64_t source_index =
              record.source_offset +
              static_cast<int64_t>(row_start) * record.source_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.source_strides[1];
          values[column] = ScaleF16x8(source + source_index, scalar_op);
        }
        Transpose8x8F16(values);
        for (uint64_t row = 0; row < tile; ++row) {
          const int64_t destination_index =
              record.destination_offset +
              static_cast<int64_t>(row_start + row) *
                  record.destination_strides[0] +
              static_cast<int64_t>(column_start) *
                  record.destination_strides[1];
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(result + destination_index),
              values[row]);
        }
        continue;
      }
      for (uint64_t row = 0; row < rows_in_tile; ++row) {
        for (uint64_t column = 0; column < columns_in_tile; ++column) {
          const int64_t source_index =
              record.source_offset +
              static_cast<int64_t>(row_start + row) *
                  record.source_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.source_strides[1];
          const int64_t destination_index =
              record.destination_offset +
              static_cast<int64_t>(row_start + row) *
                  record.destination_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.destination_strides[1];
          AssignMappedValue{}(
              result[destination_index], scalar_op(source[source_index]));
        }
      }
    }
  }
}

template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_F16C_TARGET
void ExecuteRank2F16ReverseAvx2(const Record& record,
                                const uint16_t* source, uint16_t* result,
                                uint64_t first_row, uint64_t row_count, ScalarOp scalar_op) {
  constexpr uint64_t tile = 8;
  const uint64_t columns = record.shape[1];
  const uint64_t row_stop = first_row + row_count;
  for (uint64_t row_start = first_row; row_start < row_stop;
       row_start += tile) {
    const uint64_t rows_in_tile = std::min(tile, row_stop - row_start);
    for (uint64_t column_start = 0; column_start < columns;
         column_start += tile) {
      const uint64_t columns_in_tile =
          std::min(tile, columns - column_start);
      if (rows_in_tile == tile && columns_in_tile == tile) {
        __m128i values[tile];
        for (uint64_t row = 0; row < tile; ++row) {
          const int64_t source_index =
              record.source_offset +
              static_cast<int64_t>(row_start + row) *
                  record.source_strides[0] +
              static_cast<int64_t>(column_start) *
                  record.source_strides[1];
          values[row] = ScaleF16x8(source + source_index, scalar_op);
        }
        Transpose8x8F16(values);
        for (uint64_t column = 0; column < tile; ++column) {
          const int64_t destination_index =
              record.destination_offset +
              static_cast<int64_t>(row_start) *
                  record.destination_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.destination_strides[1];
          _mm_storeu_si128(
              reinterpret_cast<__m128i*>(result + destination_index),
              values[column]);
        }
        continue;
      }
      for (uint64_t row = 0; row < rows_in_tile; ++row) {
        for (uint64_t column = 0; column < columns_in_tile; ++column) {
          const int64_t source_index =
              record.source_offset +
              static_cast<int64_t>(row_start + row) *
                  record.source_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.source_strides[1];
          const int64_t destination_index =
              record.destination_offset +
              static_cast<int64_t>(row_start + row) *
                  record.destination_strides[0] +
              static_cast<int64_t>(column_start + column) *
                  record.destination_strides[1];
          AssignMappedValue{}(
              result[destination_index], scalar_op(source[source_index]));
        }
      }
    }
  }
}
#endif

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_FMA_TARGET
__m256 FullComplexMultiplyFourC64Avx2(__m256 value,
                                      float scale_real,
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
void ExecuteScaledC64Avx2Range(const std::complex<float>* input,
                               std::complex<float>* output,
                               uint64_t element_count, float scale_real,
                               float scale_imaginary) {
  uint64_t element = 0;
  for (; element + 4 <= element_count; element += 4) {
    const __m256 value = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element));
    const __m256 mapped =
        FullComplexMultiplyFourC64Avx2(
            value, scale_real, scale_imaginary);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element), mapped);
  }
  for (; element < element_count; ++element) {
    const std::complex<float> value = input[element];
    AssignMappedValue{}(
        output[element],
        FullComplexMultiply(scale_real, scale_imaginary, value.real(),
                            value.imag()));
  }
}



template <typename ScalarOp>
TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteRank2C64Avx2Range(
    const Record& record, const std::complex<float>* source,
    std::complex<float>* result, LeafKernel kind, uint64_t unit_start,
    uint64_t unit_count, ScalarOp scalar_op) {
  constexpr bool kUnitScale = std::is_same_v<ScalarOp, IdentityScalar>;
  constexpr uint64_t kTile = 16;
  const std::size_t outer_axis =
      kind == LeafKernel::kRank2ForwardTile ? 1 : 0;
  const std::size_t inner_axis = 1 - outer_axis;
  const uint64_t outer_stop = unit_start + unit_count;
  const uint64_t inner_extent = record.shape[inner_axis];
  const auto factor = [&] {
    if constexpr (kUnitScale) return std::complex<float>{1, 0};
    else return scalar_op.factor;
  }();
  const float scale_real = factor.real();
  const float scale_imaginary = factor.imag();
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
              values[outer_lane] = FullComplexMultiplyFourC64Avx2(
                  value, scale_real, scale_imaginary);
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
              const std::complex<float> value = source[source_address];
              result[destination_address] =
                  FullComplexMultiply(scale_real, scale_imaginary,
                                      value.real(), value.imag());
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
            const std::complex<float> value = source[source_address];
            result[destination_address] =
                FullComplexMultiply(scale_real, scale_imaginary,
                                    value.real(), value.imag());
          }
        }
      }
    }
  }
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteContiguousF32C64Avx2(const float* input,
                                 std::complex<float>* output,
                                 uint64_t element_count, F32C64Scale scalar_op) {
  const float scale_real =
      scalar_op.factor.real();
  const float scale_imaginary =
      scalar_op.factor.imag();
  const __m256 scale_real8 = _mm256_set1_ps(scale_real);
  const __m256 scale_imaginary8 = _mm256_set1_ps(scale_imaginary);

  uint64_t element = 0;
  for (; element + 8 <= element_count; element += 8) {
    const __m256 values = _mm256_loadu_ps(input + element);
    const __m256 real = _mm256_mul_ps(scale_real8, values);
    const __m256 imaginary = _mm256_mul_ps(scale_imaginary8, values);
    const __m256 low = _mm256_unpacklo_ps(real, imaginary);
    const __m256 high = _mm256_unpackhi_ps(real, imaginary);
    const __m256 first = _mm256_permute2f128_ps(low, high, 0x20);
    const __m256 second = _mm256_permute2f128_ps(low, high, 0x31);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element), first);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element + 4), second);
  }
  const std::complex<float> scale{scale_real, scale_imaginary};
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], PromotedComplexMultiply(scale, input[element]));
  }
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
__m256 ComplexProductRealEightC64Avx2(
    __m256 first_value, __m256 second_value, __m256 scale_real,
    __m256 scale_imaginary) {
  const __m256 interleaved_real =
      _mm256_shuffle_ps(first_value, second_value, 0x88);
  const __m256 interleaved_imaginary =
      _mm256_shuffle_ps(first_value, second_value, 0xDD);
  const __m256 input_real = _mm256_castpd_ps(_mm256_permute4x64_pd(
      _mm256_castps_pd(interleaved_real), 0xD8));
  const __m256 input_imaginary = _mm256_castpd_ps(_mm256_permute4x64_pd(
      _mm256_castps_pd(interleaved_imaginary), 0xD8));
  return _mm256_fmadd_ps(
      scale_real, input_real,
      _mm256_xor_ps(
          _mm256_mul_ps(scale_imaginary, input_imaginary),
          _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MIN))));
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteContiguousC64F32TransposeAvx2(
    const std::complex<float>* input, float* output,
    uint64_t element_count, ProjectedComplexProduct<float> scalar_op) {
  const float scale_real =
      scalar_op.factor.real();
  const float scale_imaginary =
      scalar_op.factor.imag();
  const __m256 scale_real8 = _mm256_set1_ps(scale_real);
  const __m256 scale_imaginary8 = _mm256_set1_ps(scale_imaginary);

  uint64_t element = 0;
  for (; element + 8 <= element_count; element += 8) {
    const __m256 first_value = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element));
    const __m256 second_value = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element + 4));
    const __m256 mapped = ComplexProductRealEightC64Avx2(
        first_value, second_value, scale_real8, scale_imaginary8);
    _mm256_storeu_ps(output + element, mapped);
  }
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], scalar_op(input[element]));
  }
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteContiguousC64F32ForwardAvx2(
    const std::complex<float>* input, float* output,
    uint64_t element_count, C64F32Scale scalar_op) {
  const float scale =
      scalar_op.factor;
  const __m128 scale4 = _mm_set1_ps(scale);
  const __m256i real_lanes = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  uint64_t element = 0;
  for (; element + 4 <= element_count; element += 4) {
    const __m256 values = _mm256_loadu_ps(
        reinterpret_cast<const float*>(input + element));
    const __m256 real8 = _mm256_permutevar8x32_ps(values, real_lanes);
    const __m128 mapped =
        _mm_mul_ps(_mm256_castps256_ps128(real8), scale4);
    _mm_storeu_ps(output + element, mapped);
  }
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], scalar_op(input[element]));
  }
}

TENSOR0_STRIDE_AVX2_FMA_TARGET
void ExecuteContiguousF32C64TransposeAvx2(
    const float* input, std::complex<float>* output,
    uint64_t element_count, EmbeddedRealProduct<float> scalar_op) {
  const float scale =
      scalar_op.factor;
  const __m256 scale8 = _mm256_set1_ps(scale);
  const __m256 zeros = _mm256_setzero_ps();
  uint64_t element = 0;
  for (; element + 8 <= element_count; element += 8) {
    const __m256 mapped =
        _mm256_mul_ps(_mm256_loadu_ps(input + element), scale8);
    const __m256 low = _mm256_unpacklo_ps(mapped, zeros);
    const __m256 high = _mm256_unpackhi_ps(mapped, zeros);
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element),
                     _mm256_permute2f128_ps(low, high, 0x20));
    _mm256_storeu_ps(reinterpret_cast<float*>(output + element + 4),
                     _mm256_permute2f128_ps(low, high, 0x31));
  }
  for (; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element], scalar_op(input[element]));
  }
}
#endif

void ExecuteScaledC64ContiguousRange(
    const std::complex<float>* input, std::complex<float>* output,
    uint64_t element_count, float scale_real, float scale_imaginary) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2Fma()) {
    ExecuteScaledC64Avx2Range(
        input, output, element_count, scale_real, scale_imaginary);
    return;
  }
#endif
  for (uint64_t element = 0; element < element_count; ++element) {
    AssignMappedValue{}(
        output[element],
        FullComplexMultiply(scale_real, scale_imaginary,
                            input[element].real(), input[element].imag()));
  }
}



template <typename ScalarOp>
void ExecuteRank2C64PortableRange(
    const Record& record, const std::complex<float>* source,
    std::complex<float>* result, LeafKernel kind, uint64_t unit_start,
    uint64_t unit_count, ScalarOp scalar_op) {
  constexpr bool kUnitScale = std::is_same_v<ScalarOp, IdentityScalar>;
  constexpr uint64_t kTile = 16;
  const std::size_t outer_axis =
      kind == LeafKernel::kRank2ForwardTile ? 1 : 0;
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
                scalar_op(source[source_address]);
          }
          source_address += record.source_strides[inner_axis];
          destination_address += record.destination_strides[inner_axis];
        }
      }
    }
  }
}

template <typename ScalarOp>
void ExecuteRank2C64Range(const Record& record, const std::complex<float>* source,
                          std::complex<float>* result, LeafKernel kind,
                          uint64_t unit_start, uint64_t unit_count, ScalarOp scalar_op) {
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (CpuSupportsAvx2Fma()) {
    ExecuteRank2C64Avx2Range(record, source, result, kind, unit_start, unit_count, scalar_op);
    return;
  }
#endif
  ExecuteRank2C64PortableRange(record, source, result, kind, unit_start, unit_count, scalar_op);
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
             scalar_op(source[record.source_offset]));
    return;
  }
  ExecuteGenericAffineRange<false>(record, source, result, 0,
                                   GenericRowCount(record), scalar_op,
                                   write_op);
}

Record SliceRecordWorkRange(const Record& record, LeafKernel kind,
                            uint64_t unit_start, uint64_t unit_count) {
  Record slice = record;
  std::size_t axis = 0;
  if (kind == LeafKernel::kRank2ForwardTile) {
    axis = 1;
  } else if (kind == LeafKernel::kRank2ReverseTile ||
             kind == LeafKernel::kRank4TwoPairTile) {
    axis = 0;
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

template <typename ScalarOp>
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
TENSOR0_STRIDE_AVX2_TARGET
#endif
bool ExecuteRank4TwoPair(const Record& record, const float* source,
                         float* result, uint64_t first_axis0,
                         uint64_t axis0_count, ScalarOp scalar_op) {
  constexpr bool kUnitScale = std::is_same_v<ScalarOp, IdentityScalar>;
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if (!CpuSupportsAvx2() || !IsRank4TwoPairCandidate(record) ||
      first_axis0 > record.shape[0] ||
      axis0_count > record.shape[0] - first_axis0) {
    return false;
  }
  const float factor = [&] {
    if constexpr (kUnitScale) return 1.0F;
    else return scalar_op.factor;
  }();
  const __m256 scale = _mm256_set1_ps(factor);
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

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
TENSOR0_STRIDE_ALWAYS_INLINE void ExecuteContiguousAffineMap(
    const SourceElement* input, ResultElement* output,
    uint64_t element_count, ScalarOp scalar_op, WriteOp write_op) {
  if constexpr (std::is_same_v<SourceElement, ResultElement> &&
                std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar> &&
                kIsAssignMappedValue<WriteOp>) {
    if (input != output) std::memcpy(output, input, element_count * sizeof(SourceElement));
    return;
  }
  if constexpr (std::is_same_v<SourceElement, std::complex<float>> &&
                std::is_same_v<ResultElement, std::complex<float>> &&
                kIsAssignMappedValue<WriteOp>) {
    if constexpr (std::is_same_v<ScalarOp, C64Scale>) {
      ObserveLeafKernel(4);
      ExecuteScaledC64ContiguousRange(
          input, output, element_count, scalar_op.factor.real(), scalar_op.factor.imag());
      return;
    }
  }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  if constexpr (
      std::is_same_v<SourceElement, float> &&
      std::is_same_v<ResultElement, std::complex<float>> &&
      std::is_same_v<std::decay_t<ScalarOp>, F32C64Scale> &&
      kIsAssignMappedValue<WriteOp>) {
    if (!disable_f32_c64_contiguous_simd_for_tests.load(
            std::memory_order_relaxed) &&
        CpuSupportsAvx2Fma()) {
      ExecuteContiguousF32C64Avx2(
          input, output, element_count, scalar_op);
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, std::complex<float>> &&
      std::is_same_v<ResultElement, float> &&
      std::is_same_v<std::decay_t<ScalarOp>, ProjectedComplexProduct<float>> &&
      kIsAssignMappedValue<WriteOp>) {
    if (CpuSupportsAvx2Fma()) {
      ExecuteContiguousC64F32TransposeAvx2(
          input, output, element_count, scalar_op);
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, std::complex<float>> &&
      std::is_same_v<ResultElement, float> &&
      std::is_same_v<std::decay_t<ScalarOp>, C64F32Scale> &&
      kIsAssignMappedValue<WriteOp>) {
    if (CpuSupportsAvx2Fma()) {
      ExecuteContiguousC64F32ForwardAvx2(
          input, output, element_count, scalar_op);
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, float> &&
      std::is_same_v<ResultElement, std::complex<float>> &&
      std::is_same_v<std::decay_t<ScalarOp>, EmbeddedRealProduct<float>> &&
      kIsAssignMappedValue<WriteOp>) {
    if (CpuSupportsAvx2Fma()) {
      ExecuteContiguousF32C64TransposeAvx2(
          input, output, element_count, scalar_op);
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, uint16_t> &&
      std::is_same_v<ResultElement, uint16_t> &&
      (std::is_same_v<std::decay_t<ScalarOp>, F16Scale> ||
       std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>) &&
      kIsAssignMappedValue<WriteOp>) {
    if (!disable_f16_f16_contiguous_simd_for_tests.load(
            std::memory_order_relaxed) &&
        CpuSupportsAvx2F16c() &&
        ExecuteContiguousF16F16Avx2(
            input, output, element_count, scalar_op)) {
      if constexpr (std::is_same_v<ScalarOp, F16Scale>) ObserveLeafKernel(1);
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, uint16_t> &&
      std::is_same_v<ResultElement, float> &&
      (std::is_same_v<std::decay_t<ScalarOp>, F16F32Scale> ||
       std::is_same_v<std::decay_t<ScalarOp>, ScalarCast<ffi::F16, ffi::F32>>) &&
      kIsAssignMappedValue<WriteOp>) {
    if (!disable_f16_f32_contiguous_simd_for_tests.load(
            std::memory_order_relaxed) &&
        CpuSupportsAvx2F16c() &&
        ExecuteContiguousF16F32Avx2(
            input, output, element_count, scalar_op)) {
      return;
    }
  }
  if constexpr (
      std::is_same_v<SourceElement, float> &&
      std::is_same_v<ResultElement, uint16_t> &&
      std::is_same_v<std::decay_t<ScalarOp>, NarrowF16Product> &&
      kIsAssignMappedValue<WriteOp>) {
    if (CpuSupportsAvx2F16c()) {
      ExecuteContiguousF32F16TransposeAvx2(
          input, output, element_count, scalar_op);
      return;
    }
  }
#endif
  ExecuteContiguousMap(input, output, element_count, scalar_op,
                       write_op);
}

template <typename SourceElement, typename ResultElement, typename ScalarOp,
          typename WriteOp>
void ExecuteAffineLeafRange(
    const Record& record, const SourceElement* source, ResultElement* result,
    LeafKernel kind, uint64_t unit_start, uint64_t unit_count,
    ScalarOp scalar_op, WriteOp write_op) {
  if (unit_count == 0) {
    return;
  }
  if (kind == LeafKernel::kContiguousRow) {
    const SourceElement* input =
        source + record.source_offset + unit_start;
    ResultElement* output =
        result + record.destination_offset + unit_start;
    ExecuteContiguousAffineMap(input, output, unit_count, scalar_op,
                               write_op);
    return;
  }
  if (kind == LeafKernel::kContiguousInnerRows) {
    const auto row_op = [&](int64_t source_base, int64_t destination_base,
                            uint64_t inner_count) {
      ExecuteContiguousAffineMap(source + source_base,
                                 result + destination_base, inner_count,
                                 scalar_op, write_op);
    };
    ExecuteGenericAffineRows(record, unit_start, unit_count, row_op);
    return;
  }
  if (kind == LeafKernel::kBroadcastContiguousRow) {
    if constexpr (kIsAssignMappedValue<WriteOp>) {
      ExecuteBroadcastContiguousRows<false>(
          record, source, result, unit_start, unit_count, scalar_op);
    } else {
      ExecuteGenericAffineRange<false>(
          record, source, result, unit_start, unit_count, scalar_op,
          write_op);
    }
    return;
  }
  if (kind == LeafKernel::kRank2ForwardTile) {
    if constexpr (kIsAssignMappedValue<WriteOp>) {
      if constexpr (
          std::is_same_v<SourceElement, std::complex<float>> &&
          std::is_same_v<ResultElement, std::complex<float>> &&
          (std::is_same_v<std::decay_t<ScalarOp>, C64Scale> ||
       std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>)) {
        ExecuteRank2C64Range(
            record, source, result, kind,
            unit_start, unit_count, scalar_op);
        if constexpr (std::is_same_v<ScalarOp, C64Scale>) ObserveLeafKernel(8);
        return;
      }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      if constexpr (
          std::is_same_v<SourceElement, uint16_t> &&
          std::is_same_v<ResultElement, uint16_t> &&
          (std::is_same_v<std::decay_t<ScalarOp>, F16Scale> ||
           std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>)) {
        if (!disable_f16_f16_contiguous_simd_for_tests.load(
                std::memory_order_relaxed) &&
            CpuSupportsAvx2F16c()) {
          ExecuteRank2F16ForwardAvx2(
              record, source, result, unit_start, unit_count, scalar_op);
          if constexpr (std::is_same_v<ScalarOp, F16Scale>) ObserveLeafKernel(2);
          return;
        }
      }
#endif
      ExecuteRank2AffineColumns(record, source, result, unit_start,
                                unit_count, scalar_op, write_op);
      return;
    }
  }
  if (kind == LeafKernel::kRank2ReverseTile) {
    if constexpr (kIsAssignMappedValue<WriteOp>) {
      if constexpr (
          std::is_same_v<SourceElement, std::complex<float>> &&
          std::is_same_v<ResultElement, std::complex<float>> &&
          (std::is_same_v<std::decay_t<ScalarOp>, C64Scale> ||
       std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>)) {
        ExecuteRank2C64Range(
            record, source, result, kind,
            unit_start, unit_count, scalar_op);
        if constexpr (std::is_same_v<ScalarOp, C64Scale>) ObserveLeafKernel(8);
        return;
      }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
      if constexpr (
          std::is_same_v<SourceElement, uint16_t> &&
          std::is_same_v<ResultElement, uint16_t> &&
          (std::is_same_v<std::decay_t<ScalarOp>, F16Scale> ||
           std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>)) {
        if (!disable_f16_f16_contiguous_simd_for_tests.load(
                std::memory_order_relaxed) &&
            CpuSupportsAvx2F16c()) {
          ExecuteRank2F16ReverseAvx2(
              record, source, result, unit_start, unit_count, scalar_op);
          if constexpr (std::is_same_v<ScalarOp, F16Scale>) ObserveLeafKernel(2);
          return;
        }
      }
#endif
      ExecuteRank2AffineRows(record, source, result, unit_start, unit_count,
                             scalar_op, write_op);
      return;
    }
  }
  if (kind == LeafKernel::kRank4TwoPairTile) {
    if constexpr (std::is_same_v<SourceElement, float> &&
                  std::is_same_v<ResultElement, float> &&
                  (std::is_same_v<std::decay_t<ScalarOp>, F32Scale> ||
                   std::is_same_v<std::decay_t<ScalarOp>, IdentityScalar>) &&
                  kIsAssignMappedValue<WriteOp>) {
      const bool executed = ExecuteRank4TwoPair(record, source, result, unit_start,
                                               unit_count, scalar_op);
      if (executed) return;
    }
  }
  if (record.rank == 0) {
    ExecuteAffineUpdateRecord(
        record, source, result, scalar_op, write_op);
    return;
  }
  if (kind == LeafKernel::kScalarRow) {
    const std::size_t fastest_axis =
        record.loop_axes_fastest_first[0];
    if (record.source_strides[fastest_axis] == 1 &&
        record.destination_strides[fastest_axis] == 1) {
      const auto row_op = [&](int64_t source_base,
                              int64_t destination_base,
                              uint64_t inner_count) {
        ExecuteContiguousAffineMap(
            source + source_base, result + destination_base,
            inner_count, scalar_op, write_op);
      };
      ExecuteGenericAffineElementRangeWithOp(
          record, unit_start, unit_count, row_op);
      return;
    }
    ExecuteGenericAffineElementRange<false>(
        record, source, result, unit_start, unit_count, scalar_op,
        write_op);
    return;
  }
  ExecuteAffineUpdateRecord(
      SliceRecordWorkRange(record, kind, unit_start, unit_count), source,
      result, scalar_op, write_op);
}

struct RecordWork {
  // Scheduling coordinates for one prepared record. Address semantics stay
  // in the record owned by PreparedAffinePlan.
  std::size_t record_index = 0;
  uint64_t chunk_begin = 0;
  uint64_t chunk_count = 0;
  uint64_t units_per_chunk = 0;
  uint64_t unit_count = 0;
  uint64_t elements_per_unit = 0;
  LeafKernel leaf_kernel = LeafKernel::kScalarRow;
};

struct GeneratedRecordProgram {
  Record record;
  std::array<uint64_t, kMaximumRank> blocks{};
  std::array<uint64_t, kMaximumRank> split_costs{};
};

std::vector<uint64_t> GeneratedIndexOrder(
    const std::vector<int64_t>& strides) {
  std::vector<uint64_t> order(strides.size(), 1);
  for (std::size_t axis = 0; axis < strides.size(); ++axis) {
    const uint64_t stride = AbsoluteStride(strides[axis]);
    if (stride == 0) {
      continue;
    }
    for (int64_t candidate : strides) {
      const uint64_t candidate_stride = AbsoluteStride(candidate);
      if (candidate_stride != 0 && candidate_stride < stride) {
        ++order[axis];
      }
    }
  }
  return order;
}

uint64_t GeneratedMemoryRegion(
    const std::vector<uint64_t>& dims,
    const std::vector<std::vector<uint64_t>>& byte_strides) {
  constexpr uint64_t kCacheLineBytes = 64;
  uint64_t memory_region = 0;
  for (const auto& strides : byte_strides) {
    uint64_t contiguous_bytes = 0;
    uint64_t cache_line_blocks = 1;
    for (std::size_t axis = 0; axis < dims.size(); ++axis) {
      if (strides[axis] < kCacheLineBytes) {
        uint64_t axis_bytes = 0;
        if (!CheckedMultiply(dims[axis] - 1, strides[axis], &axis_bytes) ||
            !CheckedAdd(contiguous_bytes, axis_bytes,
                        &contiguous_bytes)) {
          return std::numeric_limits<uint64_t>::max();
        }
      } else if (!CheckedMultiply(cache_line_blocks, dims[axis],
                                  &cache_line_blocks)) {
        return std::numeric_limits<uint64_t>::max();
      }
    }
    uint64_t contiguous_lines = contiguous_bytes / kCacheLineBytes + 1;
    uint64_t operand_bytes = 0;
    if (!CheckedMultiply(kCacheLineBytes, contiguous_lines,
                         &operand_bytes) ||
        !CheckedMultiply(operand_bytes, cache_line_blocks,
                         &operand_bytes) ||
        !CheckedAdd(memory_region, operand_bytes, &memory_region)) {
      return std::numeric_limits<uint64_t>::max();
    }
  }
  return memory_region;
}

std::vector<uint64_t> ComputeGeneratedBlocks(
    const std::vector<uint64_t>& dims,
    const std::vector<uint64_t>& costs,
    const std::vector<std::vector<uint64_t>>& byte_strides,
    const std::vector<std::vector<uint64_t>>& stride_orders) {
  constexpr uint64_t kBlockMemoryBytes = UINT64_C(1) << 15;
  if (dims.empty() ||
      GeneratedMemoryRegion(dims, byte_strides) <= kBlockMemoryBytes) {
    return dims;
  }

  uint64_t minimum_order = std::numeric_limits<uint64_t>::max();
  for (const auto& orders : stride_orders) {
    minimum_order = std::min(
        minimum_order, *std::min_element(orders.begin(), orders.end()));
  }
  bool common_fastest_axis = true;
  for (const auto& orders : stride_orders) {
    common_fastest_axis =
        common_fastest_axis && orders.front() == minimum_order;
  }
  if (common_fastest_axis) {
    std::vector<uint64_t> tail_dims(dims.begin() + 1, dims.end());
    std::vector<uint64_t> tail_costs(costs.begin() + 1, costs.end());
    std::vector<std::vector<uint64_t>> tail_byte_strides;
    std::vector<std::vector<uint64_t>> tail_stride_orders;
    tail_byte_strides.reserve(byte_strides.size());
    tail_stride_orders.reserve(stride_orders.size());
    for (const auto& strides : byte_strides) {
      tail_byte_strides.emplace_back(strides.begin() + 1, strides.end());
    }
    for (const auto& orders : stride_orders) {
      tail_stride_orders.emplace_back(orders.begin() + 1, orders.end());
    }
    std::vector<uint64_t> result{dims.front()};
    const auto tail = ComputeGeneratedBlocks(
        tail_dims, tail_costs, tail_byte_strides, tail_stride_orders);
    result.insert(result.end(), tail.begin(), tail.end());
    return result;
  }

  uint64_t minimum_byte_stride = std::numeric_limits<uint64_t>::max();
  for (const auto& strides : byte_strides) {
    minimum_byte_stride = std::min(
        minimum_byte_stride,
        *std::min_element(strides.begin(), strides.end()));
  }
  if (minimum_byte_stride > kBlockMemoryBytes) {
    return std::vector<uint64_t>(dims.size(), 1);
  }

  std::vector<uint64_t> blocks = dims;
  const auto split_axis = [&]() {
    std::size_t selected = 0;
    uint64_t selected_score = 0;
    for (std::size_t axis = 0; axis < blocks.size(); ++axis) {
      uint64_t score = 0;
      if (!CheckedMultiply(blocks[axis] - 1, costs[axis], &score)) {
        score = std::numeric_limits<uint64_t>::max();
      }
      if (score >= selected_score) {
        selected = axis;
        selected_score = score;
      }
    }
    return selected;
  };
  while (GeneratedMemoryRegion(blocks, byte_strides) >=
         2 * kBlockMemoryBytes) {
    const std::size_t axis = split_axis();
    blocks[axis] = (blocks[axis] + 1) >> 1;
  }
  while (GeneratedMemoryRegion(blocks, byte_strides) >
         kBlockMemoryBytes) {
    const std::size_t axis = split_axis();
    if (blocks[axis] <= 1) {
      break;
    }
    --blocks[axis];
  }
  return blocks;
}

GeneratedRecordProgram CompileGeneratedRecord(const Record& source,
                                               uint64_t mode,
                                               uint64_t source_item_size,
                                               uint64_t result_item_size) {
  GeneratedRecordProgram program;
  program.record = source;
  const std::size_t rank = source.rank;
  std::vector<int64_t> destination_strides(
      source.destination_strides.begin(),
      source.destination_strides.begin() + rank);
  std::vector<int64_t> source_strides(
      source.source_strides.begin(), source.source_strides.begin() + rank);
  const auto destination_order = GeneratedIndexOrder(destination_strides);
  const auto source_order = GeneratedIndexOrder(source_strides);
  std::vector<std::size_t> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);
  if ((mode & kGeneratedStridedOrder) != 0) {
    constexpr uint64_t kImportanceBits = 2;
    std::stable_sort(axes.begin(), axes.end(), [&](std::size_t left,
                                                  std::size_t right) {
      const auto importance = [&](std::size_t axis) {
        const uint64_t destination =
            UINT64_C(2) << (kImportanceBits * (rank - destination_order[axis]));
        const uint64_t input =
            UINT64_C(1) << (kImportanceBits * (rank - source_order[axis]));
        return source.shape[axis] > 1 ? destination + input : 0;
      };
      return importance(left) > importance(right);
    });
  } else {
    std::copy(source.loop_axes_fastest_first.begin(),
              source.loop_axes_fastest_first.begin() + rank, axes.begin());
  }

  std::vector<uint64_t> dims(rank);
  std::vector<uint64_t> costs(rank);
  std::vector<std::vector<uint64_t>> byte_strides(2,
                                                   std::vector<uint64_t>(rank));
  std::vector<std::vector<uint64_t>> stride_orders(2,
                                                   std::vector<uint64_t>(rank));
  const auto byte_stride = [](uint64_t stride, uint64_t item_size) {
    uint64_t bytes = 0;
    return CheckedMultiply(stride, item_size, &bytes)
               ? bytes
               : std::numeric_limits<uint64_t>::max();
  };
  for (std::size_t position = 0; position < rank; ++position) {
    const std::size_t axis = axes[position];
    program.record.shape[position] = source.shape[axis];
    program.record.source_strides[position] = source.source_strides[axis];
    program.record.destination_strides[position] =
        source.destination_strides[axis];
    program.record.loop_axes_fastest_first[position] = position;
    dims[position] = source.shape[axis];
    const uint64_t source_stride = AbsoluteStride(source.source_strides[axis]);
    const uint64_t destination_stride =
        AbsoluteStride(source.destination_strides[axis]);
    const uint64_t minimum_stride =
        std::min(source_stride, destination_stride);
    costs[position] =
        minimum_stride == 0
            ? 1
            : std::min<uint64_t>(minimum_stride,
                                 std::numeric_limits<uint64_t>::max() / 2) *
                  2;
    byte_strides[0][position] =
        byte_stride(destination_stride, result_item_size);
    byte_strides[1][position] =
        byte_stride(source_stride, source_item_size);
    stride_orders[0][position] = destination_order[axis];
    stride_orders[1][position] = source_order[axis];
  }
  const std::vector<uint64_t> blocks =
      (mode & kGeneratedBlocking) != 0
          ? ComputeGeneratedBlocks(dims, costs, byte_strides, stride_orders)
          : dims;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    program.blocks[axis] = blocks[axis];
    program.split_costs[axis] = costs[axis];
  }
  return program;
}

template <std::size_t Axis, std::size_t Rank, typename Function>
void ForEachGeneratedBlockRank(
    const GeneratedRecordProgram& program,
    std::array<uint64_t, kMaximumRank>* starts, Function function) {
  const uint64_t step = std::max<uint64_t>(1, program.blocks[Axis]);
  for (uint64_t start = 0; start < program.record.shape[Axis];
       start += step) {
    (*starts)[Axis] = start;
    if constexpr (Axis == 0) {
      Record block = program.record;
      block.logical_elements = 1;
      for (std::size_t axis = 0; axis < Rank; ++axis) {
        block.source_offset +=
            static_cast<int64_t>((*starts)[axis]) *
            program.record.source_strides[axis];
        block.destination_offset +=
            static_cast<int64_t>((*starts)[axis]) *
            program.record.destination_strides[axis];
        block.shape[axis] = std::min(
            std::max<uint64_t>(1, program.blocks[axis]),
            program.record.shape[axis] - (*starts)[axis]);
        block.logical_elements *= block.shape[axis];
      }
      function(block);
    } else {
      ForEachGeneratedBlockRank<Axis - 1, Rank>(program, starts, function);
    }
  }
}

template <std::size_t Rank, typename Function>
void DispatchGeneratedBlocks(const GeneratedRecordProgram& program,
                             Function function) {
  std::array<uint64_t, kMaximumRank> starts{};
  ForEachGeneratedBlockRank<Rank - 1, Rank>(program, &starts, function);
}

template <typename Function>
void ForEachGeneratedBlock(const GeneratedRecordProgram& program,
                           Function function) {
  bool full_domain = true;
  for (std::size_t axis = 0; axis < program.record.rank; ++axis) {
    if (program.blocks[axis] != program.record.shape[axis]) {
      full_domain = false;
      break;
    }
  }
  if (full_domain) {
    function(program.record);
    return;
  }
  if (program.record.rank == 0) {
    function(program.record);
    return;
  }
#define TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(Rank) \
  case Rank:                                           \
    DispatchGeneratedBlocks<Rank>(program, function);  \
    return
  switch (program.record.rank) {
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(1);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(2);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(3);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(4);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(5);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(6);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(7);
    TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE(8);
    default:
      return;
  }
#undef TENSOR0_STRIDE_GENERATED_BLOCK_RANK_CASE
}

void AppendGeneratedSubdomains(const GeneratedRecordProgram& program,
                               uint64_t thread_budget,
                               std::vector<GeneratedRecordProgram>* output) {
  constexpr uint64_t kMinimumThreadElements = UINT64_C(1) << 15;
  if (thread_budget <= 1 ||
      program.record.logical_elements <= kMinimumThreadElements ||
      program.record.rank == 0) {
    output->push_back(program);
    return;
  }
  std::size_t split_axis = 0;
  uint64_t split_score = 0;
  for (std::size_t axis = 0; axis < program.record.rank; ++axis) {
    uint64_t score = 0;
    if (!CheckedMultiply(program.record.shape[axis] - 1,
                         program.split_costs[axis], &score)) {
      score = std::numeric_limits<uint64_t>::max();
    }
    if (score >= split_score) {
      split_axis = axis;
      split_score = score;
    }
  }
  const uint64_t dimension = program.record.shape[split_axis];
  if (program.split_costs[split_axis] == 0 ||
      dimension <= std::min<uint64_t>(program.blocks[split_axis], 1024)) {
    output->push_back(program);
    return;
  }

  const uint64_t left_extent = dimension >> 1;
  if (left_extent == 0) {
    output->push_back(program);
    return;
  }
  GeneratedRecordProgram left = program;
  GeneratedRecordProgram right = program;
  left.record.shape[split_axis] = left_extent;
  right.record.shape[split_axis] = dimension - left_extent;
  left.record.logical_elements =
      program.record.logical_elements / dimension * left_extent;
  right.record.logical_elements =
      program.record.logical_elements - left.record.logical_elements;
  right.record.source_offset +=
      static_cast<int64_t>(left_extent) *
      program.record.source_strides[split_axis];
  right.record.destination_offset +=
      static_cast<int64_t>(left_extent) *
      program.record.destination_strides[split_axis];
  left.blocks[split_axis] =
      std::min(left.blocks[split_axis], left.record.shape[split_axis]);
  right.blocks[split_axis] =
      std::min(right.blocks[split_axis], right.record.shape[split_axis]);
  const uint64_t left_threads = thread_budget >> 1;
  AppendGeneratedSubdomains(left, left_threads, output);
  AppendGeneratedSubdomains(right, thread_budget - left_threads, output);
}

constexpr uint64_t kParallelChunkBytes = UINT64_C(256) * 1024;
// Parallel work is classified only by whether it is a pure compact copy.
// Thresholds use estimated input-plus-output traffic rather than one-side
// element bytes. A plain memcpy-like stream needs more work to amortize the
// XLA thread-pool boundary than mapped or non-contiguous affine execution.
constexpr uint64_t kParallelCompactCopyMinimumWorkBytes =
    UINT64_C(8) * 1024 * 1024;
constexpr uint64_t kParallelMappedMinimumWorkBytes =
    UINT64_C(1) * 1024 * 1024;

bool IsBroadcastContiguousRowCandidate(const Record& record) {
  if (record.rank == 0) {
    return false;
  }
  const std::size_t fastest_axis = record.loop_axes_fastest_first[0];
  return record.shape[fastest_axis] > 1 &&
         record.source_strides[fastest_axis] == 0 &&
         record.destination_strides[fastest_axis] == 1;
}

bool IsContiguousInnerRowsCandidate(const Record& record) {
  if (record.rank == 0) {
    return false;
  }
  const std::size_t fastest_axis = record.loop_axes_fastest_first[0];
  return record.shape[fastest_axis] > 1 &&
         record.source_strides[fastest_axis] == 1 &&
         record.destination_strides[fastest_axis] == 1;
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

bool IsTrivialCompactCopy(const ParsedPlan& plan, bool dynamic_scale) {
  if (dynamic_scale ||
      plan.scalar_kind != kScalarForwardScaleCast ||
      plan.source_dtype != plan.result_dtype || plan.records.size() != 1 ||
      plan.coverage != kCoverageCompleteUnique ||
      plan.source_size != plan.output_size ||
      plan.copied_elements != plan.source_size) {
    return false;
  }
  const Record& record = plan.records[0];
  return record.source_offset == 0 && record.destination_offset == 0 &&
         IsCompactSameMapping(record) && record.identity_scale;
}

uint64_t ParallelMinimumWorkBytes(const ParsedPlan& plan,
                                  bool dynamic_scale = false) {
  return IsTrivialCompactCopy(plan, dynamic_scale)
             ? kParallelCompactCopyMinimumWorkBytes
             : kParallelMappedMinimumWorkBytes;
}

bool Rank2LeafSupportsDtype(uint64_t dtype) {
  if (dtype == kDtypeC64) {
    return true;
  }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
  return dtype == kDtypeF16 &&
         !disable_f16_f16_contiguous_simd_for_tests.load(
             std::memory_order_relaxed) &&
         CpuSupportsAvx2F16c();
#else
  return false;
#endif
}

ffi::Error CompileRecordWork(const ParsedPlan& plan, uint64_t item_size,
                           std::vector<RecordWork>* work,
                           uint64_t* chunks_per_batch,
                           bool force_generic = false,
                           bool disable_layout_specializations = false) {
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
    if (!force_generic && IsCompactSameMapping(record)) {
      item.leaf_kernel = LeafKernel::kContiguousRow;
      item.unit_count = record.logical_elements;
    } else if (!force_generic &&
               IsBroadcastContiguousRowCandidate(record)) {
      item.leaf_kernel = LeafKernel::kBroadcastContiguousRow;
      elements_per_unit =
          record.shape[record.loop_axes_fastest_first[0]];
      item.unit_count = GenericRowCount(record);
    } else if (!force_generic && IsContiguousInnerRowsCandidate(record)) {
      item.leaf_kernel = LeafKernel::kContiguousInnerRows;
      elements_per_unit =
          record.shape[record.loop_axes_fastest_first[0]];
      item.unit_count = GenericRowCount(record);
    } else if (!force_generic && !disable_layout_specializations &&
               Rank2LeafSupportsDtype(plan.dtype) &&
               IsRank2ForwardCandidate(record)) {
      item.leaf_kernel = LeafKernel::kRank2ForwardTile;
      item.unit_count = record.shape[1];
      elements_per_unit = record.shape[0];
    } else if (!force_generic && !disable_layout_specializations &&
               Rank2LeafSupportsDtype(plan.dtype) &&
               IsRank2ReverseCandidate(record)) {
      item.leaf_kernel = LeafKernel::kRank2ReverseTile;
      item.unit_count = record.shape[0];
      elements_per_unit = record.shape[1];
    } else if (!force_generic && !disable_layout_specializations &&
               plan.dtype == kDtypeF32 &&
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
               CpuSupportsAvx2() &&
#else
               false &&
#endif
               IsRank4TwoPairCandidate(record) &&
               record.shape[0] != 0) {
      item.leaf_kernel = LeafKernel::kRank4TwoPairTile;
      item.unit_count = record.shape[0];
      elements_per_unit =
          record.logical_elements / record.shape[0];
    } else if (record.rank == 0) {
      item.unit_count = record.logical_elements;
    } else {
      const uint64_t inner_elements =
          record.shape[record.loop_axes_fastest_first[0]];
      if (inner_elements <= target_elements) {
        elements_per_unit = inner_elements;
        item.unit_count = GenericRowCount(record);
      } else {
        item.unit_count = record.logical_elements;
      }
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

ffi::Error CompileAffinePlan(
    ffi::Span<const uint8_t> descriptor,
    ParsedPlan* prepared_plan,
    std::vector<RecordWork>* prepared_work,
    uint64_t* prepared_chunks_per_batch,
    uint64_t* prepared_parallel_minimum_bytes,
    bool dynamic_scale = false) {
  const DescriptorReader reader(descriptor);
  ParsedPlan plan;
  ffi::Error error = ParseAndValidateDescriptor(reader, &plan);
  if (!error.success()) {
    return error;
  }

  const uint64_t item_size = DtypeItemSize(plan.dtype);
  std::vector<RecordWork> work;
  uint64_t chunks_per_batch = 0;
  error = CompileRecordWork(plan, item_size, &work, &chunks_per_batch);
  if (!error.success()) {
    return error;
  }
  *prepared_plan = std::move(plan);
  *prepared_work = std::move(work);
  *prepared_chunks_per_batch = chunks_per_batch;
  *prepared_parallel_minimum_bytes =
      ParallelMinimumWorkBytes(*prepared_plan, dynamic_scale);
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


struct PreparedAffinePlan {
  // The only retained ordinary affine execution plan. CompileAffinePlan
  // canonicalizes semantic records, selects leaf opcodes, and partitions work
  // once during FFI instantiation.
  ParsedPlan plan;
  std::vector<RecordWork> work;
  uint64_t chunks_per_batch = 0;
  uint64_t parallel_minimum_bytes = 0;
  bool force_generic = false;
  uint64_t strided_generated_mode = 0;
  std::vector<GeneratedRecordProgram> strided_generated_records;
};
// Generic affine walkers, leaf kernels, generated loops, and CPU plan compilation.
