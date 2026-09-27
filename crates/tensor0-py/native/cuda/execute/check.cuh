#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace tensor0::stride::cuda {

inline void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA ") + operation + ": " + cudaGetErrorString(status));
  }
}

}  // namespace tensor0::stride::cuda
