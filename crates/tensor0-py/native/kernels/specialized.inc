namespace kernels {

#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
template <typename SourceOp>
__attribute__((target("avx2")))
void ExecuteRank4TwoPair(
    const layout::Record& record, const std::array<std::size_t, 4>& axes,
    const float* source, float* result, const SourceOp& source_op) {
  const auto [axis0, axis1, axis2, axis3] = axes;
  for (uint64_t outer0 = 0; outer0 < record.shape[axis0]; ++outer0) {
    const auto source0 = record.source_offset + static_cast<int64_t>(outer0) * record.source_strides[axis0];
    const auto destination0 = record.destination_offset + static_cast<int64_t>(outer0) * record.destination_strides[axis0];
    for (uint64_t outer1 = 0; outer1 < record.shape[axis1]; ++outer1) {
      const auto source1 = source0 + static_cast<int64_t>(outer1) * record.source_strides[axis1];
      const auto destination1 = destination0 + static_cast<int64_t>(outer1) * record.destination_strides[axis1];
      for (uint64_t column = 0; column < record.shape[axis3]; column += 8) {
        for (uint64_t row = 0; row < record.shape[axis2]; row += 8) {
          __m256 values[8];
          for (uint64_t lane = 0; lane < 8; ++lane) {
            values[lane] = _mm256_loadu_ps(source + source1 + static_cast<int64_t>(column + lane) *
                record.source_strides[axis3] + static_cast<int64_t>(row));
            if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::F32>>) {
              values[lane] = _mm256_mul_ps(values[lane], _mm256_set1_ps(source_op.factor));
            }
          }
          ::tensor0::stride::simd::Transpose8x8F32(values);
          for (uint64_t lane = 0; lane < 8; ++lane) {
            _mm256_storeu_ps(result + destination1 + static_cast<int64_t>(row + lane) *
                record.destination_strides[axis2] + static_cast<int64_t>(column), values[lane]);
          }
        }
      }
    }
  }
}

template <typename SourceOp>
__attribute__((target("avx2,fma")))
void ExecuteRank2C64Transpose(
    const std::complex<float>* source, std::complex<float>* result,
    uint64_t rows, uint64_t columns, int64_t source_stride,
    int64_t destination_stride, const SourceOp& source_op) {
  constexpr uint64_t tile = 16;
  for (uint64_t column_start = 0; column_start < columns;) {
    const auto column_stop = column_start + std::min(tile, columns - column_start);
    for (uint64_t row_start = 0; row_start < rows;) {
      const auto row_stop = row_start + std::min(tile, rows - row_start);
      uint64_t column = column_start;
      for (; column_stop - column >= 4; column += 4) {
        uint64_t row = row_start;
        for (; row_stop - row >= 4; row += 4) {
          __m256 values[4];
          for (uint64_t lane = 0; lane < 4; ++lane) {
            const auto source_index = static_cast<int64_t>(row) +
                static_cast<int64_t>(column + lane) * source_stride;
            values[lane] = _mm256_loadu_ps(reinterpret_cast<const float*>(source + source_index));
            if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
              values[lane] = ::tensor0::stride::simd::FullComplexMultiplyFourC64(
                  values[lane], source_op.factor.real(), source_op.factor.imag());
            }
          }
          ::tensor0::stride::simd::TransposeFourC64(values);
          for (uint64_t lane = 0; lane < 4; ++lane) {
            const auto destination_index = static_cast<int64_t>(row + lane) * destination_stride +
                static_cast<int64_t>(column);
            _mm256_storeu_ps(reinterpret_cast<float*>(result + destination_index), values[lane]);
          }
        }
        for (; row < row_stop; ++row) {
          for (uint64_t lane = 0; lane < 4; ++lane) {
            const auto source_index = static_cast<int64_t>(row) +
                static_cast<int64_t>(column + lane) * source_stride;
            const auto destination_index = static_cast<int64_t>(row) * destination_stride +
                static_cast<int64_t>(column + lane);
            result[destination_index] = source_op(source[source_index]);
          }
        }
      }
      for (; column < column_stop; ++column) {
        for (uint64_t row = row_start; row < row_stop; ++row) {
          const auto source_index = static_cast<int64_t>(row) + static_cast<int64_t>(column) * source_stride;
          const auto destination_index = static_cast<int64_t>(row) * destination_stride + static_cast<int64_t>(column);
          result[destination_index] = source_op(source[source_index]);
        }
      }
      row_start = row_stop;
    }
    column_start = column_stop;
  }
}

template <typename SourceOp>
__attribute__((target("avx2,f16c")))
void ExecuteRank2F16Transpose(
    const uint16_t* source, uint16_t* result, uint64_t rows, uint64_t columns,
    int64_t source_stride, int64_t destination_stride, const SourceOp& source_op) {
  constexpr uint64_t tile = 8;
  for (uint64_t column_start = 0; column_start < columns;) {
    const auto column_count = std::min(tile, columns - column_start);
    for (uint64_t row_start = 0; row_start < rows;) {
      const auto row_count = std::min(tile, rows - row_start);
      if (row_count == tile && column_count == tile) {
        __m128i values[tile];
        for (uint64_t column = 0; column < tile; ++column) {
          const auto source_index = static_cast<int64_t>(row_start) +
              static_cast<int64_t>(column_start + column) * source_stride;
          values[column] = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(source + source_index));
          if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::F16>>) {
            const auto factor = _mm256_set1_ps(
                scalar::Convert<scalar::F32, scalar::F16>(source_op.factor));
            values[column] = _mm256_cvtps_ph(
                _mm256_mul_ps(factor, _mm256_cvtph_ps(values[column])),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          }
        }
        ::tensor0::stride::simd::Transpose8x8F16(values);
        for (uint64_t row = 0; row < tile; ++row) {
          const auto destination_index =
              static_cast<int64_t>(row_start + row) * destination_stride +
              static_cast<int64_t>(column_start);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(result + destination_index), values[row]);
        }
      } else {
        for (uint64_t row = 0; row < row_count; ++row) {
          for (uint64_t column = 0; column < column_count; ++column) {
            const auto source_index = static_cast<int64_t>(row_start + row) +
                static_cast<int64_t>(column_start + column) * source_stride;
            const auto destination_index =
                static_cast<int64_t>(row_start + row) * destination_stride +
                static_cast<int64_t>(column_start + column);
            result[destination_index] = source_op(source[source_index]);
          }
        }
      }
      row_start += row_count;
    }
    column_start += column_count;
  }
}
#endif

template <typename SourceOp>
void ExecuteRank2AffineColumns(
    const layout::Record& record,
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<typename SourceOp::OutputDtype>* result,
    const SourceOp& source_op) {
  constexpr uint64_t tile = 32;
  const uint64_t rows = record.shape[0];
  const uint64_t column_stop_limit = record.shape[1];
  for (uint64_t column_start = 0;
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
          result[destination_index] = source_op(source[source_index]);
          source_index += record.source_strides[0];
          destination_index += record.destination_strides[0];
        }
      }
    }
  }
}

template <typename SourceOp>
void ExecuteRank2AffineRows(
    const layout::Record& record,
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<typename SourceOp::OutputDtype>* result,
    const SourceOp& source_op) {
  constexpr uint64_t tile = 32;
  const uint64_t columns = record.shape[1];
  const uint64_t row_stop_limit = record.shape[0];
  for (uint64_t row_start = 0; row_start < row_stop_limit;
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
          result[destination_index] = source_op(source[source_index]);
          source_index += record.source_strides[1];
          destination_index += record.destination_strides[1];
        }
      }
    }
  }
}

template <typename SourceOp>
inline void ExecuteContiguousMap(
    const scalar::Value<typename SourceOp::InputDtype>* input,
    scalar::Value<typename SourceOp::OutputDtype>* output,
    uint64_t element_count, const SourceOp& source_op) {
  uint64_t element = 0;
  if constexpr (std::is_same_v<SourceOp,
                    expression::Cast<scalar::F32, expression::Identity<scalar::F16>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousF16F32(
        input, output, element_count);
  } else if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::F16>> ||
                       std::is_same_v<SourceOp,
                           expression::Scale<scalar::F32, expression::Identity<scalar::F16>>>) {
    const float factor = [&] {
      if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::F16>>) {
        return scalar::Convert<scalar::F32, scalar::F16>(source_op.factor);
      } else {
        return source_op.factor;
      }
    }();
    element = ::tensor0::stride::simd::ExecuteContiguousF16Scale(
        input, output, element_count, factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Cast<scalar::F16, expression::Scale<scalar::F32>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousF32ProductF16(
        input, output, element_count, source_op.map.factor);
  } else if constexpr (std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousC64Scale(
        input, output, element_count, source_op.factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Scale<scalar::C64, expression::Identity<scalar::F32>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousF32C64Scale(
        input, output, element_count, source_op.factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Cast<scalar::C64, expression::Scale<scalar::F32>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousF32ProductComplex(
        input, output, element_count, source_op.map.factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Cast<scalar::F32, expression::Scale<scalar::C64>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousC64ProductReal(
        input, output, element_count, source_op.map.factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Scale<scalar::F32,
                               expression::Cast<scalar::F32, expression::Identity<scalar::C64>>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousC64RealScale(
        input, output, element_count, source_op.factor);
  } else if constexpr (std::is_same_v<SourceOp,
                           expression::Cast<scalar::F32,
                               expression::Scale<scalar::F32, expression::Identity<scalar::C64>>>>) {
    element = ::tensor0::stride::simd::ExecuteContiguousC64RealScale(
        input, output, element_count, source_op.map.factor);
  }
  for (; element < element_count; ++element) {
    output[element] = source_op(input[element]);
  }
}

}
