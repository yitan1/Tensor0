#include "../execute/dot.h"
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
          if constexpr (scalar::kIsComplex<Left>) {
            if (conjugate_left) {
              return ExecuteDot<Scalar>(thread_pool, prepared->records, left_data, right_data, result_data,
                  left_size, right_size, batch_count, expression::Conjugate<Left>{}, expression::Identity<Right>{});
            }
          }
          return ExecuteDot<Scalar>(thread_pool, prepared->records, left_data, right_data, result_data,
              left_size, right_size, batch_count, expression::Identity<Left>{}, expression::Identity<Right>{});
        };
      });
    });
  });
  if (error.failure()) return CompletedFuture(error);
  return execute();
}

}

#define TENSOR0_STRIDE_DEFINE_DOT(Suffix, Dtype) \
XLA_FFI_DEFINE_HANDLER_SYMBOL( \
    Tensor0StrideDot##Suffix##V1, tensor0::stride::Dot<ffi::Dtype>, \
    ffi::Ffi::BindExecute().Attr<ffi::Span<const int64_t>>("layout") \
        .Attr<int64_t>("conjugate_left") \
        .Ctx<ffi::State<tensor0::stride::PreparedState>>() \
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>() \
        .Ret<ffi::BufferR1<ffi::Dtype>>().Ctx<ffi::ThreadPool>()); \
extern "C" void* Tensor0StrideDot##Suffix##V1Handler() { \
  return reinterpret_cast<void*>(&Tensor0StrideDot##Suffix##V1); \
}

TENSOR0_STRIDE_FOR_EACH_DTYPE(TENSOR0_STRIDE_DEFINE_DOT)

#undef TENSOR0_STRIDE_DEFINE_DOT
