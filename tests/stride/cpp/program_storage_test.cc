#include "execute/scheduling.h"
#include <cassert>
#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace n = tensor0::stride;
static std::atomic<int> allocations{0};
static std::atomic<int> frees{0};

[[gnu::noinline]] void* operator new(std::size_t size) {
  void* p = std::malloc(size);
  if (!p) throw std::bad_alloc();
  ++allocations;
  return p;
}
[[gnu::noinline]] void operator delete(void* p) noexcept { if (p) ++frees; std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
[[gnu::noinline]] void* operator new(std::size_t size, std::align_val_t alignment) {
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(alignment), size)) throw std::bad_alloc();
  ++allocations;
  return p;
}
[[gnu::noinline]] void operator delete(void* p, std::align_val_t) noexcept { ::operator delete(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(p, alignment);
}

struct Tracker {
  int* alive;
  explicit Tracker(int* alive) : alive(alive) { ++*alive; }
  ~Tracker() { --*alive; }
};
struct alignas(128) Aligned {
  Tracker tracker;
  explicit Aligned(int* alive) : tracker(alive) {}
};
struct Throwing {
  Tracker tracker;
  explicit Throwing(int* alive) : tracker(alive) { throw 42; }
};

template <typename State>
void Check() {
  int alive = 0;
  const auto before = allocations.load();
  const auto freed = frees.load();
  auto typed = n::MakeProgramState<State>(&alive);
  assert(allocations == before + 1 && frees == freed && alive == 1);
  assert(reinterpret_cast<uintptr_t>(typed.get()) % alignof(State) == 0);
  std::shared_ptr<const void> erased = typed;
  std::weak_ptr<const void> weak = erased;
  typed.reset();
  assert(alive == 1 && !weak.expired());
  erased.reset();
  assert(alive == 0 && weak.expired() && frees == freed);
  weak.reset();
  assert(frees == freed + 1);
}

int main() {
  Check<Tracker>();
  Check<Aligned>();
  int alive = 0;
  const auto before = allocations.load();
  const auto freed = frees.load();
  try {
    n::MakeProgramState<Throwing>(&alive);
    assert(false);
  } catch (int value) {
    assert(value == 42);
  }
  assert(alive == 0 && allocations == before + 1 && frees == freed + 1);
}
