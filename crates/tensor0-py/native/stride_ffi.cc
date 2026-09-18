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
#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/construction.inc"
#include "layout/planning.inc"
#include "layout/blocking.inc"
#include "ffi/dtype.inc"
#include "ffi/descriptor.inc"
#include "runtime/runtime.inc"
#include "kernels/affine.inc"
#include "kernels/reduction.inc"
#include "execute/map.inc"
#include "execute/update.inc"
#include "execute/map_tasks.inc"
#include "execute/reduction.inc"
#include "execute/dot.inc"
#include "execute/reduction_tasks.inc"

}

#include "ffi/bindings.inc"
