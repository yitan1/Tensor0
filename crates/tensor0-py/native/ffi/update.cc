#include "../execute/update.h"
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

struct SelectedUpdateCoefficient {
  ffi::DataType bound_type;
  UpdateCoefficientReader reader;
};

template <typename Scalar>
constexpr ffi::DataType UpdateScalarType() {
#define TENSOR0_UPDATE_SCALAR_TYPE(Suffix, Dtype) \
  if constexpr (std::is_same_v<Scalar, ScalarDtype<ffi::Dtype>>) return ffi::Dtype;
  TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_UPDATE_SCALAR_TYPE)
#undef TENSOR0_UPDATE_SCALAR_TYPE
}

template <typename Mapped>
SelectedUpdateCoefficient SelectUpdateCoefficient(
    ffi::AnyBuffer buffer, uint64_t count, const void* result, uint64_t output_bytes) {
  SelectedUpdateCoefficient selected;
  VisitCoefficient(buffer, [&](auto raw_type, const auto* data) {
    using Raw = typename decltype(raw_type)::type;
    using Bound = UpdateCoefficient<Raw, Mapped>;
    ValidateDisjointBuffers(data, BufferBytes(count, sizeof(scalar::Value<Raw>)),
                            result, output_bytes);
    selected = {UpdateScalarType<Bound>(), {data, count, ReadUpdateCoefficient<Raw, Bound>}};
  });
  return selected;
}

// Only the image of the existing coefficient policy reaches the bound executor.
template <typename Mapped, typename Function>
void VisitBoundUpdateCoefficient(ffi::DataType type, Function function) {
  VisitScalarDtype(type, [&](auto bound_type) {
    using Bound = typename decltype(bound_type)::type;
    if constexpr (std::is_same_v<Bound, UpdateCoefficient<Bound, Mapped>>) {
      function(bound_type);
    }
  });
}

// Invocation borrows buffers and records only until the synchronous executor entry returns.
struct UpdateInvocation {
  ffi::ThreadPool thread_pool;
  const PreparedState* prepared;
  const void* source;
  const void* base;
  void* result;
  uint64_t source_size;
  uint64_t output_size;
  uint64_t batch_count;
  UpdateCoefficientReader alpha;
  UpdateCoefficientReader beta;
};

template <typename Result, typename Source, typename Alpha, typename Beta>
ffi::Future InvokeUpdate(const UpdateInvocation& request) {
  return ExecuteBoundUpdate<Result, Alpha, Beta>(
      request.thread_pool, request.prepared->records,
      static_cast<const scalar::Value<Source>*>(request.source),
      static_cast<const scalar::Value<Result>*>(request.base),
      static_cast<scalar::Value<Result>*>(request.result),
      request.source_size, request.output_size, request.batch_count,
      request.alpha, request.beta,
      expression::Identity<Source>{}, expression::Identity<Result>{});
}

template <ffi::DataType Dtype>
ffi::Future Update(ffi::Span<const int64_t>, const PreparedState* prepared,
                         ffi::AnyBuffer source,
                         ffi::BufferR2<Dtype> base,
                         ffi::AnyBuffer alpha,
                         ffi::AnyBuffer beta,
                         ffi::ResultBufferR2<Dtype> result, ffi::ThreadPool thread_pool) {
  using Scalar = ScalarDtype<Dtype>;
  UpdateInvocation invocation{thread_pool, prepared, nullptr, base.typed_data(), result->typed_data(),
      0, 0, 0, {}, {}};
  ffi::Future (*execute)(const UpdateInvocation&) = nullptr;
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
        const auto alpha_reader = SelectUpdateCoefficient<Source>(
            alpha, alpha_count, result->typed_data(), output_bytes);
        const auto beta_reader = SelectUpdateCoefficient<Scalar>(
            beta, beta_count, result->typed_data(), output_bytes);
        VisitBoundUpdateCoefficient<Source>(alpha_reader.bound_type, [&](auto alpha_type) {
          using Alpha = typename decltype(alpha_type)::type;
          VisitBoundUpdateCoefficient<Scalar>(beta_reader.bound_type, [&](auto beta_type) {
            using Beta = typename decltype(beta_type)::type;
            invocation.source = source_data;
            invocation.source_size = source_size;
            invocation.output_size = output_size;
            invocation.batch_count = batch_count;
            invocation.alpha = alpha_reader.reader;
            invocation.beta = beta_reader.reader;
            execute = &InvokeUpdate<Scalar, Source, Alpha, Beta>;
          });
        });
      } else {
        throw std::invalid_argument("unsupported update source/storage dtype pair");
      }
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute(invocation);
}

}

#define TENSOR0_STRIDE_DEFINE_UPDATE(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideUpdate##Suffix##V1, tensor0::stride::Update<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::BufferR2<ffi::Dtype>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>() \
        .Ret<ffi::BufferR2<ffi::Dtype>>().Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideUpdate##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideUpdate##Suffix##V1); \
}

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_UPDATE)

#undef TENSOR0_STRIDE_DEFINE_UPDATE
