#pragma once

#include "xla/ffi/api/ffi.h"
#include <cstdint>

namespace tensor0::stride::cuda {

struct BoundCoefficient {
  const void* data;
  xla::ffi::DataType dtype;
  uint64_t count;
};

}  // namespace tensor0::stride::cuda
