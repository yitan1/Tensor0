#include <cstdlib>
extern "C" {
int cudaGetLastError() { std::abort(); }
void* __cudaRegisterFatBinary(void*) { std::abort(); }
void __cudaRegisterFunction() { std::abort(); }
void __cudaRegisterVar() { std::abort(); }
void __cudaRegisterFatBinaryEnd() { std::abort(); }
void __cudaUnregisterFatBinary() { std::abort(); }
}
