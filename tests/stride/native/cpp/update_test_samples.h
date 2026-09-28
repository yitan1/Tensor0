#pragma once

#include "numeric/scalar.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

namespace update_test {
namespace scalar = tensor0::stride::scalar;
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

}
