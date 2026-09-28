#include "execute/dot.h"
#include "numeric/expression.h"
#include <cassert>
#include <type_traits>

namespace n = tensor0::stride;
namespace s = n::scalar;
namespace e = n::expression;

using Input = n::DotInput<e::Identity<s::F32>, e::Identity<s::F32>>;
using State = n::Program<Input, n::SumReduction, n::ZeroReduction>;

template <typename Accumulator>
void Check(const std::shared_ptr<const void>& owner, const Input& input,
           const std::vector<n::layout::GeneratedRecordProgram>& records) {
  auto program = n::MakeReductionProgram<Accumulator>(input);
  // All output dtypes accept the exact same stored input-state type.
  program.state = owner;
  s::Value<Accumulator> actual = s::Convert<Accumulator, s::S32>(17);
  s::Value<Accumulator> expected{};
  n::ExecuteDotBatch<Accumulator>(records, input.left, input.right, &expected,
                                  input.left_op, input.right_op);
  program.entry->initialize(program.state.get(), &actual, 1);
  for (const auto& record : records) {
    program.entry->execute(program.state.get(), record, 0, &actual);
  }
  assert(actual == expected);
}

int main() {
  static_assert(std::is_same_v<s::Value<s::U16>, s::Value<s::F16>>);
  static_assert(std::is_same_v<s::Value<s::U16>, s::Value<s::BF16>>);
  const float left[]{1.5f, 2.25f}, right[]{2.0f, 2.0f};
  const Input input{left, 2, right, 2, {}, {}};
  auto owner = n::MakeProgramState<State>(n::ZeroReduction{}, input);
  std::weak_ptr<const void> weak = owner;
  const auto record = n::layout::BuildLayout({2}, {1}, 0, {1}, 0, 2, 2, 0);
  const auto records = n::layout::PrepareGeneratedRecords({record}, 4, 4, false);
  Check<s::F32>(owner, input, records);
  Check<s::F64>(owner, input, records);
  Check<s::S64>(owner, input, records);
  Check<s::U16>(owner, input, records);
  Check<s::F16>(owner, input, records);
  Check<s::BF16>(owner, input, records);
  Check<s::C64>(owner, input, records);
  assert(owner.use_count() == 1);
  owner.reset();
  assert(weak.expired());
}
