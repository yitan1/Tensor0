#include "stride_descriptor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <complex>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TENSOR0_STRIDE_HAS_AVX2_TARGET 1
#define TENSOR0_STRIDE_AVX2_TARGET __attribute__((target("avx2")))
#define TENSOR0_STRIDE_AVX2_FMA_TARGET __attribute__((target("avx2,fma")))
#define TENSOR0_STRIDE_AVX2_F16C_TARGET \
  __attribute__((target("avx2,f16c")))
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TENSOR0_STRIDE_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define TENSOR0_STRIDE_ALWAYS_INLINE inline
#endif

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifndef TENSOR0_STRIDE_JAX_VERSION
#define TENSOR0_STRIDE_JAX_VERSION "unknown"
#endif

#ifndef TENSOR0_STRIDE_JAXLIB_VERSION
#define TENSOR0_STRIDE_JAXLIB_VERSION "unknown"
#endif

namespace ffi = xla::ffi;

namespace tensor0::stride {
namespace {

// Keep the implementation in one translation unit so its templates and
// anonymous prepared-state types do not become a separate internal ABI.
#include "stride_common.inc"
#include "stride_scalar.inc"
#include "stride_affine.inc"
#include "stride_reduction_plan.inc"
#include "stride_runtime.inc"
#include "stride_reduction.inc"

}  // namespace
}  // namespace tensor0::stride

#include "stride_bindings.inc"
