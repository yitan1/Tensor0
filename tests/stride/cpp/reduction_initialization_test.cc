#include "execute/reduction_program.h"
#include "ffi/dtype.h"
#include <array>
#include <cassert>
namespace n = tensor0::stride;
namespace s = n::scalar;
namespace ffi = xla::ffi;

struct First {
  template <typename A>
  void operator()(const n::layout::GeneratedRecordProgram&, uint64_t,
                  s::Value<A>*, std::type_identity<A>) const {}
};
struct Second : First { uint64_t payload = 37; };

template <typename A>
void Check() {
  auto a = n::MakeReductionProgram<A>(First{});
  auto b = n::MakeReductionProgram<A>(Second{});
  assert(a.entry->initialize == b.entry->initialize);
  using V = s::Value<A>;
  for (auto* program : {&a, &b}) {
    const auto one = s::Convert<A, s::F32>(1.0f);
    std::array<V,4> values{one,one,one,one};
    program->entry->initialize(program->state.get(),values.data()+1,2);
    assert(values[0]==one && values[3]==one);
    assert(values[1]==V{} && values[2]==V{});
    program->entry->initialize(program->state.get(),values.data(),0);
    assert(values[0]==one && values[3]==one);
  }
}
int main() {
#define CHECK(Suffix, Dtype) Check<n::ScalarDtype<ffi::Dtype>>();
  TENSOR0_STRIDE_FOR_EACH_DTYPE(CHECK)
#undef CHECK
}
