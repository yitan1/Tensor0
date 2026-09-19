namespace tensor0::stride {

std::atomic<uint64_t> prepared_created_count{0};
std::atomic<uint64_t> prepared_destroyed_count{0};

struct PreparedState {
  static ffi::TypeId id;

  explicit PreparedState(descriptor::DecodedLayout value)
      : records(std::move(value.records)), source_size(value.source_size), output_size(value.output_size) {
    prepared_created_count.fetch_add(1, std::memory_order_relaxed);
  }

  PreparedState(const PreparedState&) = delete;
  PreparedState& operator=(const PreparedState&) = delete;

  ~PreparedState() {
    prepared_destroyed_count.fetch_add(1, std::memory_order_relaxed);
  }

  const std::vector<layout::Record> records;
  const uint64_t source_size;
  const uint64_t output_size;
};

ffi::TypeId PreparedState::id = {};
const ffi::TypeInfo kPreparedTypeInfo = ffi::MakeTypeInfo<PreparedState>();

template <typename Execute>
ffi::Error ContainErrors(Execute execute) {
  try {
    execute();
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(std::string("tensor0-native: ") + error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("tensor0-native: ") + error.what());
  } catch (...) {
    return ffi::Error::Internal("tensor0-native: unknown exception");
  }
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

template <ffi::DataType Dtype>
ffi::Future Copy(ffi::Span<const int64_t>, const PreparedState* prepared,
                ffi::AnyBuffer source,
                ffi::ResultBufferR2<Dtype> result, ffi::ThreadPool thread_pool) {
  using Scalar = ScalarDtype<Dtype>;
  std::function<ffi::Future()> execute;
  uint64_t batch_count = 0;
  const uint64_t source_size = prepared->source_size;
  const uint64_t output_size = prepared->output_size;
  const auto error = ContainErrors([&] {
    if (source.dimensions().size() != 2) {
      throw std::invalid_argument("copy storage must be rank-two");
    }
    for (const auto dimensions : {source.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("copy dimensions must be nonnegative");
      }
    }
    if (source.dimensions()[0] != result->dimensions()[0]) {
      throw std::invalid_argument("copy batch dimensions do not match");
    }
    batch_count = source.dimensions()[0];
    ValidatePreparedDimensions(*prepared, source.dimensions()[1], result->dimensions()[1]);
    uint64_t source_elements = 0;
    uint64_t output_elements = 0;
    if (!layout::CheckedMultiply(batch_count, source_size, &source_elements) ||
        !layout::CheckedMultiply(batch_count, output_size, &output_elements)) {
      throw std::invalid_argument("copy batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(output_elements, sizeof(scalar::Value<Scalar>));
    VisitScalarDtype(source.element_type(), [&](auto type) {
      using Source = typename decltype(type)::type;
      const auto* data = source.reinterpret_data<scalar::Value<Source>>();
      ValidateDisjointBuffers(data, BufferBytes(source_elements, sizeof(*data)),
                             result->typed_data(), output_bytes);
      execute = [=, result_data = result->typed_data()] {
        return ExecuteCopy<Source, Scalar>(thread_pool, prepared->records,
            data, result_data, source_size, output_size, batch_count);
      };
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute();
}

template <ffi::DataType Dtype>
ffi::Future Update(ffi::Span<const int64_t>, const PreparedState* prepared,
                         ffi::AnyBuffer source,
                         ffi::BufferR2<Dtype> base,
                         ffi::AnyBuffer alpha,
                         ffi::AnyBuffer beta,
                         ffi::ResultBufferR2<Dtype> result, ffi::ThreadPool thread_pool) {
  using Scalar = ScalarDtype<Dtype>;
  std::function<ffi::Future()> execute;
  uint64_t batch_count = 0;
  uint64_t output_size = 0;
  const auto error = ContainErrors([&] {
    if (source.dimensions().size() != 2) {
      throw std::invalid_argument("update source storage must be rank-two");
    }
    for (const auto dimensions : {source.dimensions(), base.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("batch dimensions must be nonnegative");
      }
    }
    if (source.dimensions()[0] != result->dimensions()[0] ||
        base.dimensions()[0] != result->dimensions()[0] ||
        base.dimensions()[1] != result->dimensions()[1]) {
      throw std::invalid_argument("update batch dimensions do not match");
    }
    batch_count = result->dimensions()[0];
    const uint64_t source_size = source.dimensions()[1];
    output_size = result->dimensions()[1];
    uint64_t source_elements = 0;
    uint64_t output_elements = 0;
    if (!layout::CheckedMultiply(batch_count, source_size, &source_elements) ||
        !layout::CheckedMultiply(batch_count, output_size, &output_elements)) {
      throw std::invalid_argument("update batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(output_elements, sizeof(scalar::Value<Scalar>));
    const auto coefficient_count = [](auto dimensions) -> uint64_t {
      if (dimensions.size() == 0) return 1;
      if (dimensions.size() != 1 || dimensions[0] < 0) {
        throw std::invalid_argument("batch coefficients must be scalars or rank-one buffers");
      }
      return dimensions[0];
    };
    const auto alpha_count = coefficient_count(alpha.dimensions());
    const auto beta_count = coefficient_count(beta.dimensions());
    if ((alpha_count != 1 && alpha_count != batch_count) ||
        (beta_count != 1 && beta_count != batch_count)) {
      throw std::invalid_argument("update coefficients must be shared or one value per batch");
    }
    if (base.typed_data() != result->typed_data()) {
      ValidateDisjointBuffers(base.typed_data(), output_bytes, result->typed_data(), output_bytes);
    }
    const auto& records = prepared->records;
    ValidatePreparedDimensions(*prepared, source_size, output_size);
    VisitScalarDtype(source.element_type(), [&](auto source_type) {
      using Source = typename decltype(source_type)::type;
      if constexpr (std::is_same_v<Source, Scalar> ||
          (std::is_same_v<Source, scalar::F16> && Dtype == ffi::F32) ||
          (std::is_same_v<Source, scalar::F32> && Dtype == ffi::F16) ||
          (std::is_same_v<Source, scalar::F32> && Dtype == ffi::C64) ||
          (std::is_same_v<Source, scalar::C64> && Dtype == ffi::F32) ||
          (std::is_same_v<Source, scalar::F64> && Dtype == ffi::C128) ||
          (std::is_same_v<Source, scalar::C128> && Dtype == ffi::F64) ||
          (std::is_same_v<Source, scalar::S32> &&
              (Dtype == ffi::PRED || Dtype == ffi::S8 || Dtype == ffi::S16)) ||
          (std::is_same_v<Source, scalar::U32> && Dtype == ffi::U8)) {
        const auto* source_data = source.reinterpret_data<scalar::Value<Source>>();
        const auto source_bytes = BufferBytes(source_elements, sizeof(*source_data));
        if (source_bytes != 0 && output_bytes != 0 &&
            static_cast<const void*>(source_data) == static_cast<const void*>(result->typed_data())) {
          if (!std::is_same_v<Source, Scalar> || source_size != output_size ||
              base.typed_data() != result->typed_data()) {
            throw std::invalid_argument("unsupported source/result alias: requires identical source/base/result storage");
          }
          for (const auto& record : records) {
            if (record.source_offset != record.destination_offset ||
                record.source_strides != record.destination_strides) {
              throw std::invalid_argument("source/result alias requires identical per-element addresses");
            }
          }
        } else {
          ValidateDisjointBuffers(source_data, source_bytes, result->typed_data(), output_bytes);
        }
        VisitCoefficient(alpha, [&](auto alpha_type, const auto* alpha_data) {
          using Alpha = typename decltype(alpha_type)::type;
          VisitCoefficient(beta, [&](auto beta_type, const auto* beta_data) {
            using Beta = typename decltype(beta_type)::type;
            ValidateDisjointBuffers(alpha_data, BufferBytes(alpha_count, sizeof(scalar::Value<Alpha>)),
                                     result->typed_data(), output_bytes);
            ValidateDisjointBuffers(beta_data, BufferBytes(beta_count, sizeof(scalar::Value<Beta>)),
                                     result->typed_data(), output_bytes);
            execute = [=, base_data = base.typed_data(), result_data = result->typed_data()] {
              return ExecuteUpdate<Scalar, Alpha, Beta>(
                  thread_pool, prepared->records, source_data, base_data, result_data,
                  source_size, output_size, batch_count,
                  alpha_data, alpha_count, beta_data, beta_count,
                  expression::Identity<Source>{}, expression::Identity<Scalar>{});
            };
          });
        });
      } else {
        throw std::invalid_argument("unsupported update source/storage dtype pair");
      }
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute();
}

template <ffi::DataType Dtype>
ffi::Future Dot(
    ffi::Span<const int64_t>, int64_t conjugate_left, const PreparedState* prepared,
    ffi::AnyBuffer left, ffi::AnyBuffer right,
    ffi::ResultBufferR1<Dtype> result, ffi::ThreadPool thread_pool) {
  using Scalar = ScalarDtype<Dtype>;
  std::function<ffi::Future()> execute;
  uint64_t batch_count = 0;
  const uint64_t left_size = prepared->source_size;
  const uint64_t right_size = prepared->output_size;
  const auto error = ContainErrors([&] {
    if (conjugate_left != 0 && conjugate_left != 1) {
      throw std::invalid_argument("dot conjugate_left must be zero or one");
    }
    if (left.dimensions().size() != 2 || right.dimensions().size() != 2) {
      throw std::invalid_argument("dot inputs must be rank-two");
    }
    for (const auto dimensions : {left.dimensions(), right.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("dot dimensions must be nonnegative");
      }
    }
    batch_count = left.dimensions()[0];
    if (static_cast<uint64_t>(right.dimensions()[0]) != batch_count ||
        static_cast<uint64_t>(result->dimensions()[0]) != batch_count ||
        static_cast<uint64_t>(left.dimensions()[1]) != left_size ||
        static_cast<uint64_t>(right.dimensions()[1]) != right_size) {
      throw std::invalid_argument("dot buffer dimensions do not match layout");
    }
    uint64_t left_elements = 0;
    uint64_t right_elements = 0;
    if (!layout::CheckedMultiply(left_size, batch_count, &left_elements) ||
        !layout::CheckedMultiply(right_size, batch_count, &right_elements)) {
      throw std::invalid_argument("dot batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(batch_count, sizeof(scalar::Value<Scalar>));
    VisitScalarDtype(left.element_type(), [&](auto left_type) {
      using Left = typename decltype(left_type)::type;
      VisitScalarDtype(right.element_type(), [&](auto right_type) {
        using Right = typename decltype(right_type)::type;
        const auto* left_data = left.reinterpret_data<scalar::Value<Left>>();
        const auto* right_data = right.reinterpret_data<scalar::Value<Right>>();
        ValidateDisjointBuffers(left_data, BufferBytes(left_elements, sizeof(*left_data)),
                               result->typed_data(), output_bytes);
        ValidateDisjointBuffers(right_data, BufferBytes(right_elements, sizeof(*right_data)),
                               result->typed_data(), output_bytes);
        execute = [=, result_data = result->typed_data()] {
          if (conjugate_left) {
            return ExecuteDot<Scalar>(thread_pool, prepared->records, left_data, right_data, result_data,
                left_size, right_size, batch_count, expression::Conjugate<Left>{}, expression::Identity<Right>{});
          } else {
            return ExecuteDot<Scalar>(thread_pool, prepared->records, left_data, right_data, result_data,
                left_size, right_size, batch_count, expression::Identity<Left>{}, expression::Identity<Right>{});
          }
        };
      });
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute();
}

template <ffi::DataType Dtype>
ffi::Future Reduce(
    ffi::Span<const int64_t> coefficient_records,
    const PreparedState* prepared, ffi::AnyBuffer source,
    ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
    ffi::ThreadPool thread_pool) {
  using Accumulator = ScalarDtype<Dtype>;
  struct CoefficientBuffer {
    ffi::DataType dtype;
    const void* data;
    bool batched;
  };
  std::function<ffi::Future()> execute;
  uint64_t batch_count = 0;
  const uint64_t source_size = prepared->source_size;
  const uint64_t output_size = prepared->output_size;
  const auto error = ContainErrors([&] {
    if (source.dimensions().size() != 2) {
      throw std::invalid_argument("reduction source storage must be rank-two");
    }
    for (const auto dimensions : {source.dimensions(), result->dimensions()}) {
      for (const auto dimension : dimensions) {
        if (dimension < 0) throw std::invalid_argument("reduction dimensions must be nonnegative");
      }
    }
    batch_count = source.dimensions()[0];
    if (batch_count != static_cast<uint64_t>(result->dimensions()[0]) ||
        static_cast<uint64_t>(source.dimensions()[1]) != prepared->source_size ||
        static_cast<uint64_t>(result->dimensions()[1]) != prepared->output_size) {
      throw std::invalid_argument("reduction buffer dimensions do not match layout");
    }
    uint64_t source_elements = 0;
    uint64_t output_elements = 0;
    if (!layout::CheckedMultiply(batch_count, prepared->source_size, &source_elements) ||
        !layout::CheckedMultiply(batch_count, prepared->output_size, &output_elements)) {
      throw std::invalid_argument("reduction batch storage size overflows");
    }
    const auto output_bytes = BufferBytes(output_elements, sizeof(scalar::Value<Accumulator>));
    if (coefficient_records.size() != coefficients.size()) {
      throw std::invalid_argument("reduction coefficient operand count does not match record indices");
    }
    std::vector<std::size_t> record_parameters(prepared->records.size(), coefficients.size());
    std::vector<CoefficientBuffer> buffers;
    buffers.reserve(coefficients.size());
    int64_t previous_record = -1;
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
      const auto record = coefficient_records[index];
      if (record <= previous_record || static_cast<uint64_t>(record) >= prepared->records.size()) {
        throw std::invalid_argument("reduction coefficient record indices must be increasing and in range");
      }
      previous_record = record;
      record_parameters[record] = index;
      const auto buffer = coefficients.get<ffi::AnyBuffer>(index);
      if (!buffer.has_value()) throw std::invalid_argument("reduction coefficient must be a buffer");
      const auto dimensions = buffer->dimensions();
      if (dimensions.size() > 1 || (dimensions.size() == 1 &&
          (dimensions[0] < 0 || (dimensions[0] != 1 &&
           static_cast<uint64_t>(dimensions[0]) != batch_count)))) {
        throw std::invalid_argument("reduction coefficients must be scalar, length-one, or match the batch count");
      }
      const uint64_t count = dimensions.size() == 0 ? 1 : dimensions[0];
      VisitCoefficient(*buffer, [&](auto type, const auto* data) {
        using Coefficient = typename decltype(type)::type;
        ValidateDisjointBuffers(data, BufferBytes(count, sizeof(scalar::Value<Coefficient>)),
                               result->typed_data(), output_bytes);
        buffers.push_back({buffer->element_type(), data, count != 1});
      });
    }
    VisitScalarDtype(source.element_type(), [&](auto type) {
      using Source = typename decltype(type)::type;
      const auto* source_data = source.reinterpret_data<scalar::Value<Source>>();
      ValidateDisjointBuffers(source_data, BufferBytes(source_elements, sizeof(*source_data)),
                             result->typed_data(), output_bytes);
      execute = [=, result_data = result->typed_data(),
                 record_parameters = std::move(record_parameters),
                 buffers = std::move(buffers)] {
        return ExecuteReduction<Source, Accumulator>(
          thread_pool, prepared->records, source_data, result_data,
          source_size, output_size, batch_count,
          [record_parameters, buffers](std::size_t record, uint64_t batch, auto apply) {
            const auto parameter = record_parameters[record];
            if (parameter == buffers.size()) {
              apply(expression::Identity<Source>{});
            } else {
              const auto& buffer = buffers[parameter];
              VisitScalarDtype(buffer.dtype, [&](auto coefficient_type) {
                using Coefficient = typename decltype(coefficient_type)::type;
                const auto* data = static_cast<const scalar::Value<Coefficient>*>(buffer.data);
                const auto factor = data[buffer.batched ? batch : 0];
                if (scalar::IsZero<Coefficient>(factor)) return;
                if (scalar::IsOne<Coefficient>(factor)) {
                  apply(expression::Identity<Source>{});
                  return;
                }
                apply(expression::Scale<Coefficient, expression::Identity<Source>>{factor, {}});
              });
            }
          });
      };
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute();
}

template <ffi::DataType Dtype>
ffi::Future Reduction(
    ffi::Span<const uint8_t>, ffi::Span<const int64_t> coefficient_records,
    const PreparedState* prepared, ffi::AnyBuffer source,
    ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
    ffi::ThreadPool thread_pool) {
  return Reduce<Dtype>(coefficient_records, prepared, source, coefficients, result, thread_pool);
}

template <ffi::DataType Dtype>
ffi::Future Accumulation(
    ffi::Span<const int64_t>, ffi::Span<const int64_t> coefficient_records,
    const PreparedState* prepared, ffi::AnyBuffer source,
    ffi::RemainingArgs coefficients, ffi::ResultBufferR2<Dtype> result,
    ffi::ThreadPool thread_pool) {
  return Reduce<Dtype>(coefficient_records, prepared, source, coefficients, result, thread_pool);
}

}
