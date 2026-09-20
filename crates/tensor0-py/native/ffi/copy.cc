#include "../execute/copy.h"
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

}

#define TENSOR0_STRIDE_DEFINE_COPY(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideCopy##Suffix##V1, tensor0::stride::Copy<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Ret<ffi::BufferR2<ffi::Dtype>>() \
        .Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideCopy##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideCopy##Suffix##V1); \
}

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_COPY)

#undef TENSOR0_STRIDE_DEFINE_COPY
