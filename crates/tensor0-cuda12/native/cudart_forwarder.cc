#include <cstring>
#include <math.h>
#include <cuda_runtime_api.h>
// The SDK compiler-internal host header uses a two-pass include protocol.
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include <crt/host_runtime.h>
#include <crt/host_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
// SDK crt/device_functions.h exposes this declaration only under __CUDACC__.
extern "C" unsigned CUDARTAPI __cudaPushCallConfiguration(dim3, dim3, size_t, cudaStream_t);

// POD state: CUDA TU constructors may call us before any C++ dynamic initializer.
static pthread_once_t once = PTHREAD_ONCE_INIT;
static int state;
static const char* failure;
static void* runtime;
#define POINTER(name) static decltype(&name) p_##name;
POINTER(cudaGetErrorString)
POINTER(cudaGetLastError)
POINTER(cudaMemcpyAsync)
POINTER(cudaMemsetAsync)
POINTER(cudaLaunchKernel)
POINTER(__cudaPushCallConfiguration)
POINTER(__cudaPopCallConfiguration)
POINTER(__cudaRegisterFatBinary)
POINTER(__cudaRegisterFatBinaryEnd)
POINTER(__cudaRegisterFunction)
POINTER(__cudaUnregisterFatBinary)
#undef POINTER
static void initialize() {
  state = -1;
  runtime = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
  if (!runtime) { failure = "resident libcudart.so.12 absent; restart required"; return; }
#define RESOLVE(name) p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(runtime, #name)); if (!p_##name) { failure = #name " absent; restart required"; return; }
  RESOLVE(cudaGetErrorString)
  RESOLVE(cudaGetLastError)
  RESOLVE(cudaMemcpyAsync)
  RESOLVE(cudaMemsetAsync)
  RESOLVE(cudaLaunchKernel)
  RESOLVE(__cudaPushCallConfiguration)
  RESOLVE(__cudaPopCallConfiguration)
  RESOLVE(__cudaRegisterFatBinary)
  RESOLVE(__cudaRegisterFatBinaryEnd)
  RESOLVE(__cudaRegisterFunction)
  RESOLVE(__cudaUnregisterFatBinary)
#undef RESOLVE
  state = 1;
}
static bool ready() {
  pthread_once(&once, initialize);
  return state == 1;
}
extern "C" {
const char* CUDARTAPI cudaGetErrorString(cudaError_t e) { return ready() ? p_cudaGetErrorString(e) : failure; }
cudaError_t CUDARTAPI cudaGetLastError() { return ready() ? p_cudaGetLastError() : cudaErrorSharedObjectSymbolNotFound; }
cudaError_t CUDARTAPI cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t t) { return ready() ? p_cudaMemcpyAsync(d,s,n,k,t) : cudaErrorSharedObjectSymbolNotFound; }
cudaError_t CUDARTAPI cudaMemsetAsync(void* d, int v, size_t n, cudaStream_t t) { return ready() ? p_cudaMemsetAsync(d,v,n,t) : cudaErrorSharedObjectSymbolNotFound; }
cudaError_t CUDARTAPI cudaLaunchKernel(const void* f, dim3 g, dim3 b, void** a, size_t n, cudaStream_t t) { return ready() ? p_cudaLaunchKernel(f,g,b,a,n,t) : cudaErrorSharedObjectSymbolNotFound; }
unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 g, dim3 b, size_t n, cudaStream_t t) { return ready() ? p___cudaPushCallConfiguration(g,b,n,t) : cudaErrorSharedObjectSymbolNotFound; }
cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3* g, dim3* b, size_t* n, void* t) { return ready() ? p___cudaPopCallConfiguration(g,b,n,t) : cudaErrorSharedObjectSymbolNotFound; }
void** CUDARTAPI __cudaRegisterFatBinary(void* f) { return ready() ? p___cudaRegisterFatBinary(f) : nullptr; }
void CUDARTAPI __cudaRegisterFatBinaryEnd(void** h) { if (ready() && h) p___cudaRegisterFatBinaryEnd(h); }
void CUDARTAPI __cudaUnregisterFatBinary(void** h) { if (ready() && h) p___cudaUnregisterFatBinary(h); }
void CUDARTAPI __cudaRegisterFunction(void** h, const char* f, char* d, const char* n, int l, uint3* tid, uint3* bid, dim3* bd, dim3* gd, int* ws) { if (ready() && h) p___cudaRegisterFunction(h,f,d,n,l,tid,bid,bd,gd,ws); }
// Fixed host-only validation ABI; no CUDA entrypoints are exported.
int Tensor0CudaForwarderStatusV1() { return ready() ? 0 : cudaErrorSharedObjectSymbolNotFound; }
void* Tensor0CudaForwarderRuntimeV1() { ready(); return runtime; }
const char* Tensor0CudaForwarderErrorV1() { ready(); return failure ? failure : "ok"; }
}
