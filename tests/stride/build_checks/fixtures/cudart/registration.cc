#include <cstring>
#include <math.h>
#include <cuda_runtime_api.h>
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include <crt/host_runtime.h>
#include <crt/host_runtime.h>
static void** binary;
__attribute__((constructor)) static void register_binary() {
  binary = __cudaRegisterFatBinary(nullptr);
  __cudaRegisterFunction(binary, nullptr, nullptr, "fixture", 0, nullptr, nullptr, nullptr, nullptr, nullptr);
#ifdef TEST_DEVICE_VAR
  __cudaRegisterVar(binary, nullptr, nullptr, "fixture_var", 0, sizeof(int), 0, 0);
#endif
  __cudaRegisterFatBinaryEnd(binary);
}
__attribute__((destructor)) static void unregister_binary() { __cudaUnregisterFatBinary(binary); }
extern "C" int Tensor0FixtureLastError() { return cudaGetLastError(); }
