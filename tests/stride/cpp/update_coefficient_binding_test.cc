#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include "layout/record.h"
#include "layout/traversal.h"
#include "layout/blocking.h"
#include "ffi/prepared.h"
#include "execute/scheduling.h"
#include "numeric/scalar.h"
#include "numeric/expression.h"
#include "kernels/generic.h"
#include "kernels/specialized.h"
#include "kernels/dispatch.h"
#include "ffi/dtype.h"
#include "execute/update.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <complex>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;












#include "thread_pool_test_support.h"

namespace native = tensor0::stride;
namespace layout = native::layout;
namespace scalar = native::scalar;
namespace expression = native::expression;

template <typename Dtype>
std::vector<scalar::Value<Dtype>> Samples() {
  std::vector<scalar::Value<Dtype>> values;
  for (double value : std::initializer_list<double>{-INFINITY, -1e30, -65504.0, -1.0001, -1.0, -0.0, 0.0,
                       1e-40, 0.0001, 0.5, 1.0, 1.0001, 2.0, 65504.0, 1e30, INFINITY, NAN}) {
    values.push_back(scalar::Convert<Dtype, scalar::F64>(value));
  }
  using Value = scalar::Value<Dtype>;
  if constexpr (std::is_integral_v<Value> && !scalar::kIsHalf<Dtype>) {
    values.push_back(std::numeric_limits<Value>::min());
    values.push_back(std::numeric_limits<Value>::max());
    if constexpr (std::numeric_limits<Value>::digits >= 54) {
      values.push_back(static_cast<Value>((UINT64_C(1) << 53) + 1));
      values.push_back(std::numeric_limits<Value>::max() - 1);
    }
  } else if constexpr (scalar::kIsComplex<Dtype>) {
    values.push_back({1, -0.0});
    values.push_back({0, 1});
    values.push_back({1, INFINITY});
    values.push_back({NAN, 1});
    values.push_back({1, NAN});
    values.push_back({INFINITY, 1});
    values.push_back({0, INFINITY});
    values.push_back({-0.0, -0.0});
    values.push_back({1, 1});
  }
  return values;
}

template <typename Dtype>
void CheckSame(scalar::Value<Dtype> actual, scalar::Value<Dtype> expected) {
  if constexpr (scalar::kIsHalf<Dtype>) {
    CheckSame<scalar::F32>(scalar::Convert<scalar::F32, Dtype>(actual),
                           scalar::Convert<scalar::F32, Dtype>(expected));
  } else if constexpr (scalar::kIsComplex<Dtype>) {
    CheckSame<scalar::Component<Dtype>>(actual.real(), expected.real());
    CheckSame<scalar::Component<Dtype>>(actual.imag(), expected.imag());
  } else if constexpr (std::is_floating_point_v<scalar::Value<Dtype>>) {
    assert((std::isnan(actual) && std::isnan(expected)) || actual == expected);
    if (actual == 0 && expected == 0) assert(std::signbit(actual) == std::signbit(expected));
  } else {
    assert(actual == expected);
  }
}

void CheckAllCoefficientProducts() {
  constexpr std::array<ffi::DataType, 15> types{
      ffi::PRED, ffi::S8, ffi::S16, ffi::S32, ffi::S64, ffi::U8, ffi::U16, ffi::U32,
      ffi::U64, ffi::F16, ffi::BF16, ffi::F32, ffi::F64, ffi::C64, ffi::C128};
  for (auto coefficient_type : types) {
    native::VisitScalarDtype(coefficient_type, [&](auto coefficient_tag) {
      using Coefficient = typename decltype(coefficient_tag)::type;
      for (auto mapped_type : types) {
        native::VisitScalarDtype(mapped_type, [&](auto mapped_tag) {
          using Mapped = typename decltype(mapped_tag)::type;
          using Bound = native::UpdateCoefficient<Coefficient, Mapped>;
          using Product = scalar::Promote<Coefficient, Mapped>;
          static_assert(std::is_same_v<Product, scalar::Promote<Bound, Mapped>>);
          static_assert(scalar::kIsComplex<Bound> == scalar::kIsComplex<Coefficient>);
          if constexpr (scalar::kIsComplex<Coefficient> || scalar::kIsHalf<Product> ||
                        std::is_integral_v<scalar::Value<Product>>) {
            static_assert(std::is_same_v<Coefficient, Bound>);
          }
          for (auto coefficient : Samples<Coefficient>()) {
            const auto bound = scalar::Convert<Bound, Coefficient>(coefficient);
            assert(scalar::IsZero<Coefficient>(coefficient) == scalar::IsZero<Bound>(bound));
            assert(scalar::IsOne<Coefficient>(coefficient) == scalar::IsOne<Bound>(bound));
            for (auto mapped : Samples<Mapped>()) {
              CheckSame<Product>(scalar::Multiply<Bound, Mapped>(bound, mapped),
                                  scalar::Multiply<Coefficient, Mapped>(coefficient, mapped));
            }
          }
        });
      }
    });
  }
}

template <typename Source, typename Result, typename Alpha, typename Beta>
void CheckBoundRecords() {
  constexpr uint64_t count = 67;
  const auto samples = Samples<Source>();
  const auto base_samples = Samples<Result>();
  std::array<scalar::Value<Source>, count> source;
  std::array<scalar::Value<Result>, count> base, result, expected;
  for (std::size_t index = 0; index < count; ++index) {
    source[index] = samples[index % samples.size()];
    base[index] = base_samples[(index + 5) % base_samples.size()];
  }
  const std::vector<layout::Record> records{
      layout::BuildLayout({count}, {1}, 0, {1}, 0, count, count, 0),
      layout::BuildLayout({count}, {-1}, count - 1, {1}, 0, count, count, 0),
      layout::BuildLayout({count}, {0}, 3, {1}, 0, count, count, 0),
      layout::BuildLayout({0}, {1}, count, {1}, count, count, count, 0)};
  for (auto alpha : Samples<Alpha>()) {
    for (auto beta : Samples<Beta>()) {
      for (const auto& record : records) {
        result = expected = base;
        // The original typed record executor remains an independent, unnormalized path.
        native::ExecuteUpdateRecord<Result, Alpha, Beta>(record, source.data(), base.data(), expected.data(),
            alpha, beta, expression::Identity<Source>{}, expression::Identity<Result>{});
        auto execute = native::BindUpdateRecord<Result, Alpha, Beta>(source.data(), base.data(), result.data(),
            alpha, beta, expression::Identity<Source>{}, expression::Identity<Result>{});
        execute(record);
        for (std::size_t index = 0; index < count; ++index) CheckSame<Result>(result[index], expected[index]);
      }
    }
  }
}

void CheckSharedCallbackTypes() {
  float source = 2, base = 3, result = 0;
  const auto first = native::BindUpdateRecord<scalar::F32, scalar::S8, scalar::S16>(
      &source, &base, &result, 2, 3, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  const auto second = native::BindUpdateRecord<scalar::F32, scalar::F16, scalar::F32>(
      &source, &base, &result, scalar::Convert<scalar::F16, scalar::F32>(2), 3,
      expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  const auto wide = native::BindUpdateRecord<scalar::F32, scalar::F64, scalar::F32>(
      &source, &base, &result, 2, 3, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  assert(first.target_type() == second.target_type());
  assert(first.target_type() != wide.target_type());
}

void CheckComplexCallbackTypes() {
  std::complex<float> source{2, 1}, base{3, -1}, result{};
  const auto integer = native::BindUpdateRecord<scalar::C64, scalar::S16, scalar::F16>(
      &source, &base, &result, 2, scalar::Convert<scalar::F16, scalar::F32>(3),
      expression::Identity<scalar::C64>{}, expression::Identity<scalar::C64>{});
  const auto real = native::BindUpdateRecord<scalar::C64, scalar::F32, scalar::S64>(
      &source, &base, &result, 2, 3,
      expression::Identity<scalar::C64>{}, expression::Identity<scalar::C64>{});
  const auto complex = native::BindUpdateRecord<scalar::C64, scalar::C64, scalar::F32>(
      &source, &base, &result, {2, 0}, 3,
      expression::Identity<scalar::C64>{}, expression::Identity<scalar::C64>{});
  assert(integer.target_type() == real.target_type());
  assert(integer.target_type() != complex.target_type());
}

int main() {
  static_assert(std::is_same_v<native::UpdateCoefficient<scalar::S64, scalar::C64>, scalar::F32>);
  static_assert(std::is_same_v<native::UpdateCoefficient<scalar::BF16, scalar::C128>, scalar::F64>);
  static_assert(std::is_same_v<native::UpdateCoefficient<scalar::F64, scalar::C64>, scalar::F64>);
  static_assert(std::is_same_v<native::UpdateCoefficient<scalar::C64, scalar::C128>, scalar::C64>);
  CheckAllCoefficientProducts();
  CheckSharedCallbackTypes();
  CheckComplexCallbackTypes();
  CheckBoundRecords<scalar::C64, scalar::C64, scalar::S64, scalar::BF16>();
  CheckBoundRecords<scalar::C64, scalar::C64, scalar::F16, scalar::F64>();
  CheckBoundRecords<scalar::C64, scalar::C64, scalar::F64, scalar::F16>();
  CheckBoundRecords<scalar::C128, scalar::C128, scalar::S64, scalar::F32>();
  CheckBoundRecords<scalar::C128, scalar::C128, scalar::F16, scalar::BF16>();
  CheckBoundRecords<scalar::C64, scalar::C64, scalar::C64, scalar::S16>();
  CheckBoundRecords<scalar::C128, scalar::C128, scalar::C64, scalar::S16>();
  CheckBoundRecords<scalar::C64, scalar::F32, scalar::S16, scalar::F32>();
  CheckBoundRecords<scalar::C128, scalar::F64, scalar::BF16, scalar::S64>();
  CheckBoundRecords<scalar::F32, scalar::C64, scalar::S16, scalar::S64>();
  CheckBoundRecords<scalar::F64, scalar::C128, scalar::U64, scalar::F16>();
  CheckBoundRecords<scalar::F32, scalar::F32, scalar::S64, scalar::BF16>();
  CheckBoundRecords<scalar::F32, scalar::F32, scalar::F64, scalar::F16>();
  CheckBoundRecords<scalar::F16, scalar::F32, scalar::BF16, scalar::S16>();
  CheckBoundRecords<scalar::BF16, scalar::BF16, scalar::F16, scalar::F16>();
  CheckBoundRecords<scalar::F64, scalar::F64, scalar::U64, scalar::F32>();
  CheckBoundRecords<scalar::S64, scalar::S64, scalar::U64, scalar::F32>();
  CheckBoundRecords<scalar::U64, scalar::U64, scalar::S64, scalar::F64>();
  CheckBoundRecords<scalar::F32, scalar::C64, scalar::S16, scalar::F32>();
  CheckBoundRecords<scalar::C64, scalar::F32, scalar::F32, scalar::S16>();
}
