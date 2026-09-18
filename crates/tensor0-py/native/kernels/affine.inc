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
inline constexpr bool kIsIdentityMapping = false;

template <typename Dtype>
inline constexpr bool kIsIdentityMapping<expression::Identity<Dtype>> = true;

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

template <std::size_t Count, typename RowOp>
void ForEachAffineRow(
    const layout::Record& record, std::array<int64_t, Count> offsets,
    const std::array<const std::vector<int64_t>*, Count>& strides, RowOp row_op) {
  const auto element_count = layout::ElementCount(record);
  if (element_count == 0) return;
  const uint64_t inner_count = record.shape.empty() ? 1 : record.shape[0];
  const uint64_t row_count = element_count / inner_count;
  std::vector<uint64_t> coordinates(record.shape.size());
  for (uint64_t row = 0; row < row_count; ++row) {
    row_op(offsets, inner_count);
    if (row + 1 == row_count) break;
    for (std::size_t axis = 1; axis < record.shape.size(); ++axis) {
      if (coordinates[axis] + 1 < record.shape[axis]) {
        ++coordinates[axis];
        for (std::size_t input = 0; input < Count; ++input) {
          offsets[input] += (*strides[input])[axis];
        }
        break;
      }
      for (std::size_t input = 0; input < Count; ++input) {
        offsets[input] -= static_cast<int64_t>(coordinates[axis]) * (*strides[input])[axis];
      }
      coordinates[axis] = 0;
    }
  }
}

template <typename SourceOp, typename Result = typename SourceOp::OutputDtype>
void ExecuteGenericAffineScalarRow(
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<Result>* result,
    int64_t source_base, int64_t destination_base, uint64_t inner_count,
    int64_t source_inner_stride, int64_t destination_inner_stride,
    const SourceOp& source_op) {
  int64_t source_index = source_base;
  int64_t destination_index = destination_base;
  for (uint64_t inner = 0; inner < inner_count; ++inner) {
    result[destination_index] =
        scalar::Convert<Result, typename SourceOp::OutputDtype>(source_op(source[source_index]));
    source_index += source_inner_stride;
    destination_index += destination_inner_stride;
  }
}

template <typename Dtype>
void ExecuteFillRecord(const layout::Record& record, scalar::Value<Dtype>* result,
                       scalar::Value<Dtype> value) {
  const int64_t inner_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<1>(record, {record.destination_offset}, {&record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t destination_index = offsets[0];
    if (inner_stride == 1) {
      std::fill_n(result + destination_index, inner_count, value);
    } else {
      for (uint64_t inner = 0; inner < inner_count; ++inner) {
        if (inner != 0) {
          destination_index += inner_stride;
        }
        result[destination_index] = value;
      }
    }
  });
}

template <typename Result, typename BinaryOp>
void ExecuteBinaryMapRecord(
    const layout::Record& record,
    const scalar::Value<typename BinaryOp::InputDtype>* source,
    const scalar::Value<typename BinaryOp::BaseInputDtype>* base,
    scalar::Value<Result>* result, const BinaryOp& operation) {
  const int64_t source_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t destination_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    int64_t source_index = offsets[0];
    int64_t destination_index = offsets[1];
    for (uint64_t inner = 0; inner < inner_count; ++inner) {
      const auto value = operation(source[source_index], base[destination_index]);
      result[destination_index] = scalar::Convert<Result, typename BinaryOp::OutputDtype>(value);
      source_index += source_stride;
      destination_index += destination_stride;
    }
  });
}

template <typename SourceOp, typename Result = typename SourceOp::OutputDtype>
void ExecuteMapRecord(
    const layout::Record& record,
    const scalar::Value<typename SourceOp::InputDtype>* source,
    scalar::Value<Result>* result,
    const SourceOp& source_op) {
  const auto element_count = layout::ElementCount(record);
  if (element_count == 0) {
    return;
  }
  if constexpr (std::is_same_v<Result, typename SourceOp::OutputDtype>) {
    if (layout::IsCompactSameMapping(record)) {
      const auto* input = source + record.source_offset;
      auto* output = result + record.destination_offset;
      if constexpr (kIsIdentityMapping<SourceOp>) {
        if (input != output) {
          std::memcpy(output, input, element_count * sizeof(*source));
        }
      } else {
        ExecuteContiguousMap(input, output, element_count, source_op);
      }
      return;
    }
#if defined(TENSOR0_STRIDE_HAS_AVX2_TARGET)
    if constexpr (std::is_same_v<SourceOp, expression::Identity<scalar::F32>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::F32>>) {
      std::array<std::size_t, 4> axes;
      if (::tensor0::stride::simd::CpuSupportsAvx2() &&
          layout::IsRank4TwoPairCandidate(record, &axes)) {
        ExecuteRank4TwoPair(record, axes, source, result, source_op);
        return;
      }
    }
    if constexpr (std::is_same_v<SourceOp, expression::Identity<scalar::F16>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::F16>> ||
                  std::is_same_v<SourceOp, expression::Identity<scalar::C64>> ||
                  std::is_same_v<SourceOp, expression::Scale<scalar::C64>>) {
      constexpr bool is_half = std::is_same_v<typename SourceOp::InputDtype, scalar::F16>;
      constexpr uint64_t vector_size = is_half ? 8 : 4;
      if (record.shape.size() == 2 && record.shape[0] >= vector_size && record.shape[1] >= vector_size &&
          (is_half ? ::tensor0::stride::simd::CpuSupportsAvx2F16c()
                   : ::tensor0::stride::simd::CpuSupportsAvx2Fma())) {
        for (std::size_t axis = 0; axis < 2; ++axis) {
          const auto other = 1 - axis;
          if (record.source_strides[axis] == 1 && record.destination_strides[other] == 1 &&
              record.source_strides[other] > 0 && record.destination_strides[axis] > 0 &&
              static_cast<uint64_t>(record.source_strides[other]) >= record.shape[axis] &&
              static_cast<uint64_t>(record.destination_strides[axis]) >= record.shape[other]) {
            if constexpr (is_half) {
              ExecuteRank2F16Transpose(source + record.source_offset, result + record.destination_offset,
                  record.shape[axis], record.shape[other], record.source_strides[other],
                  record.destination_strides[axis], source_op);
            } else {
              ExecuteRank2C64Transpose(source + record.source_offset, result + record.destination_offset,
                  record.shape[axis], record.shape[other], record.source_strides[other],
                  record.destination_strides[axis], source_op);
            }
            return;
          }
        }
      }
    }
#endif
    if (layout::IsRank2ForwardCandidate(record)) {
      ExecuteRank2AffineColumns(record, source, result, source_op);
      return;
    }
    if (layout::IsRank2ReverseCandidate(record)) {
      ExecuteRank2AffineRows(record, source, result, source_op);
      return;
    }
  }
  const int64_t source_stride = record.shape.empty() ? 0 : record.source_strides[0];
  const int64_t destination_stride = record.shape.empty() ? 0 : record.destination_strides[0];
  ForEachAffineRow<2>(record, {record.source_offset, record.destination_offset},
                     {&record.source_strides, &record.destination_strides},
                     [&](const auto& offsets, uint64_t inner_count) {
    const int64_t source_base = offsets[0];
    const int64_t destination_base = offsets[1];
    if (inner_count > 1 && source_stride == 0 && destination_stride == 1) {
      const auto mapped = scalar::Convert<Result, typename SourceOp::OutputDtype>(
          source_op(source[source_base]));
      std::fill_n(result + destination_base, inner_count, mapped);
    } else if constexpr (std::is_same_v<Result, typename SourceOp::OutputDtype>) {
      if (source_stride == 1 && destination_stride == 1) {
        ExecuteContiguousMap(source + source_base, result + destination_base, inner_count, source_op);
      } else {
        ExecuteGenericAffineScalarRow<SourceOp, Result>(
            source, result, source_base, destination_base, inner_count,
            source_stride, destination_stride, source_op);
      }
    } else {
      if (source_stride == 1 && destination_stride == 1) {
        ExecuteContiguousMap(source + source_base, result + destination_base, inner_count,
            expression::Cast<Result, SourceOp>{source_op});
      } else {
        ExecuteGenericAffineScalarRow<SourceOp, Result>(
            source, result, source_base, destination_base, inner_count,
            source_stride, destination_stride, source_op);
      }
    }
  });
}

}
