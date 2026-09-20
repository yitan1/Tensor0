#include "../execute/reduction.h"
#include "../numeric/expression.h"
#include "dtype.h"
#include "errors.h"
#include "prepared.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ffi = xla::ffi;

namespace tensor0::stride {

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

#define TENSOR0_STRIDE_DEFINE_REDUCTION(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideReduction##Suffix##V1, tensor0::stride::Reduction<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const uint8_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().RemainingArgs().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideReduction##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideReduction##Suffix##V1); \
}

#define TENSOR0_STRIDE_DEFINE_ACCUMULATION(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideAccumulation##Suffix##V1, tensor0::stride::Accumulation<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<ffi::Span<const int64_t>>("coefficient_records") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().RemainingArgs().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideAccumulation##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideAccumulation##Suffix##V1); \
}

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_REDUCTION)
TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_ACCUMULATION)

#undef TENSOR0_STRIDE_DEFINE_REDUCTION
#undef TENSOR0_STRIDE_DEFINE_ACCUMULATION
