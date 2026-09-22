#pragma once

#include "reduction_partials.h"
#include <type_traits>

namespace tensor0::stride {

// The stack context stays alive until ordinary traversal returns or throws.
template <typename Function>
void ExecuteReductionBlocks(const layout::GeneratedRecordProgram& program, const Function& function) {
  ExecuteGeneratedBlocks(program, &function, [](const void* context, const layout::Record& block) {
    (*static_cast<const Function*>(context))(block);
  });
}


// These policies describe the existing kernels; arbitrary reductions are not supported.
struct SumReduction {};

struct ZeroReduction {
  template <typename Value>
  void operator()(Value* target, uint64_t count) const {
    std::fill_n(target, count, Value{});
  }
};

template <typename Accumulator>
struct ReductionProgramEntry {
  void (*initialize)(const void*, scalar::Value<Accumulator>*, uint64_t);
  void (*execute)(const void*, const layout::GeneratedRecordProgram&, uint64_t,
                  scalar::Value<Accumulator>*);
};

// Zero initialization depends on the accumulator, not the mapping expression.
template <typename Accumulator>
void InitializeReductionTarget(const void*, scalar::Value<Accumulator>* target, uint64_t count) {
  ZeroReduction{}(target, count);
}

template <typename Accumulator, typename F>
OwnedProgram<ReductionProgramEntry<Accumulator>> MakeReductionProgram(F f) {
  using State = Program<F, SumReduction, ZeroReduction>;
  static const ReductionProgramEntry<Accumulator> entry{
      InitializeReductionTarget<Accumulator>,
      [](const void* state, const auto& program, uint64_t batch, auto* target) {
        // The dtype tag restores arithmetic type, including distinct dtypes sharing storage.
        static_cast<const State*>(state)->f(program, batch, target, std::type_identity<Accumulator>{});
      }};
  return {MakeProgramState<State>(ZeroReduction{}, std::move(f)), &entry};
}

// Only accumulator storage specializes task management. F is restored at a record boundary.
template <typename Accumulator>
ffi::Future ExecuteOwnedReductionProgram(
    ffi::ThreadPool thread_pool, ReductionPlan plan,
    scalar::Value<Accumulator>* result, uint64_t output_size, uint64_t batch_count,
    OwnedProgram<ReductionProgramEntry<Accumulator>> program);

}
