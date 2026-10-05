#include "ffi/prepared.h"
#include "execute/reduction.h"
#include "numeric/expression.h"

#include <array>
#include <atomic>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>

namespace native = tensor0::stride;

int main() {
  native::descriptor::DecodedLayout decoded;
  decoded.source_size = 8;
  decoded.output_size = 5;
  decoded.records = {
      {0, 5, 2, {3}, {-1}, {0}},
      {1, 5, 1, {3}, {0}, {1}},
      {2, 0, 0, {0}, {1}, {1}}};
  using Programs = std::vector<native::layout::GeneratedRecordProgram>;
  std::weak_ptr<const Programs> lifetime;
  {
    native::PreparedState state(std::move(decoded));
    std::array<std::shared_ptr<const Programs>, 12> first;
    std::vector<std::thread> threads;
    for (size_t index = 0; index < first.size(); ++index) {
      threads.emplace_back([&, index] { first[index] = state.AccumulationPrograms(8, 8); });
    }
    for (auto& thread : threads) thread.join();
    const auto f64 = first.front();
    for (const auto& item : first) assert(item == f64);
    assert(f64 == state.AccumulationPrograms(8, 8));
    const auto f32 = state.AccumulationPrograms(4, 4);
    const auto mixed = state.AccumulationPrograms(4, 8);
    assert(f32 != f64 && mixed != f64 && f32 != mixed);
    assert(f32 == state.AccumulationPrograms(4, 4));
    assert(mixed == state.AccumulationPrograms(4, 8));

    const auto reference = native::layout::PrepareGeneratedRecords(state.records, 8, 8, false);
    std::atomic<bool> good{true};
    threads.clear();
    for (int t = 0; t < 12; ++t) {
      threads.emplace_back([&, t] {
        const auto programs = state.AccumulationPrograms(8, 8);
        if (programs != f64) good = false;
        const std::array<double, 8> source{1, 2, 3, 4, 5, 6, 7, 8};
        std::array<double, 5> output{}, expected{};
        for (int repeat = 0; repeat < 6; ++repeat) {
          const double factor = repeat % 3 == 0 ? 0 : (t + repeat + 1) * 0.25;
          const auto bind = [&](size_t index, auto apply) {
            if (index == 1 && factor == 0) return;
            apply(native::expression::Scale<native::scalar::F64>{index == 1 ? factor : 1.0});
          };
          native::ExecuteReductionBatch<native::scalar::F64, native::scalar::F64>(
              *programs, source.data(), output.data(), output.size(), bind);
          native::ExecuteReductionBatch<native::scalar::F64, native::scalar::F64>(
              reference, source.data(), expected.data(), expected.size(), bind);
          if (output != expected || output != std::array<double, 5>{0, 6 * factor,
                  15 + 6 * factor, 6 * factor, 0}) good = false;
        }
      });
    }
    for (auto& thread : threads) thread.join();
    assert(good);
    lifetime = f64;
    first.fill(nullptr);
    // The state and remaining caller references still own the cached program.
    assert(!lifetime.expired());
  }
  assert(lifetime.expired());
}
