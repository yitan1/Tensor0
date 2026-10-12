#include <cstring>
#include <math.h>
#include <cuda_runtime_api.h>
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#include <crt/host_runtime.h>
#include <crt/host_runtime.h>
#include <cstdio>
#include <cstdlib>
static int token[2], registrations, ends, functions, variables, unregistrations;
static void valid(void** h) { if (h != reinterpret_cast<void**>(&token[0]) && h != reinterpret_cast<void**>(&token[1])) std::abort(); }
extern "C" {
const char* CUDARTAPI cudaGetErrorString(cudaError_t) { return "fake diagnostic only"; }
#ifndef MISSING_NORMAL
cudaError_t CUDARTAPI cudaGetLastError() { return static_cast<cudaError_t>(123); }
#endif
cudaError_t CUDARTAPI cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaMemsetAsync(void*, int, size_t, cudaStream_t) { return cudaSuccess; }
cudaError_t CUDARTAPI cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t) { std::abort(); }
unsigned CUDARTAPI __cudaPushCallConfiguration(dim3, dim3, size_t, cudaStream_t) { std::abort(); }
cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3*, dim3*, size_t*, void*) { std::abort(); }
void** CUDARTAPI __cudaRegisterFatBinary(void*) { if (registrations >= 2) std::abort(); return reinterpret_cast<void**>(&token[registrations++]); }
void CUDARTAPI __cudaRegisterFatBinaryEnd(void** h) { valid(h); ends++; }
void CUDARTAPI __cudaUnregisterFatBinary(void** h) { valid(h); unregistrations++; }
#ifndef MISSING_REGISTRATION
void CUDARTAPI __cudaRegisterFunction(void** h, const char*, char*, const char*, int, uint3*, uint3*, dim3*, dim3*, int*) { valid(h); functions++; }
#endif
void CUDARTAPI __cudaRegisterVar(void** h, char*, char*, const char*, int, size_t, int, int) { valid(h); variables++; }
}
__attribute__((destructor)) static void report() { fprintf(stderr,"FAKE registrations=%d ends=%d functions=%d variables=%d unregistrations=%d\n",registrations,ends,functions,variables,unregistrations); }
