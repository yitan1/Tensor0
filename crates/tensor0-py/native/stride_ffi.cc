#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kernels/avx2.inc"

#include "xla/ffi/api/ffi.h"

#ifndef TENSOR0_STRIDE_JAX_VERSION
#define TENSOR0_STRIDE_JAX_VERSION "unknown"
#endif

#ifndef TENSOR0_STRIDE_JAXLIB_VERSION
#define TENSOR0_STRIDE_JAXLIB_VERSION "unknown"
#endif

namespace ffi = xla::ffi;

namespace tensor0::stride {

#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/record.inc"
#include "layout/traversal.inc"
#include "layout/blocking.inc"
#include "ffi/dtype.inc"
#include "ffi/descriptor.inc"
#include "execute/scheduling.inc"
#include "kernels/generic.inc"
#include "kernels/specialized.inc"
#include "kernels/dispatch.inc"
#include "execute/map.inc"
#include "execute/reduction.inc"

}

#include "ffi/handlers.inc"
#include "ffi/bindings.inc"
