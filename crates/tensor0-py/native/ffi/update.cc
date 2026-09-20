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
