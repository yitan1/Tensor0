#pragma once

#include "scheduling.h"
#include "../kernels/dispatch.h"
#include "../layout/blocking.h"
#include "../numeric/expression.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor0::stride {

// Preserve uncovered output storage; an in-place batch needs no initialization.
template <typename Result>
void InitializeUpdateOutput(const scalar::Value<Result>* base,
                            scalar::Value<Result>* result, uint64_t output_size) {
  if (output_size != 0 && result != base) {
    std::memcpy(result, base, output_size * sizeof(scalar::Value<Result>));
  }
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
void ExecuteUpdateRecord(
    const layout::Record& record,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<typename BaseMap::InputDtype>* base,
    scalar::Value<Result>* result,
    scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  if (layout::ElementCount(record) == 0) return;
  using ScaledSource = expression::Scale<Alpha, SourceMap>;
  using ScaledBase = expression::Scale<Beta, BaseMap>;
  if (scalar::IsZero<Alpha>(alpha)) {
    if (scalar::IsZero<Beta>(beta)) {
      kernels::ExecuteFillRecord<Result>(record, result, {});
    } else {
      auto base_record = record;
      base_record.source_offset = record.destination_offset;
      base_record.source_strides = record.destination_strides;
      if (scalar::IsOne<Beta>(beta)) {
        kernels::ExecuteMapRecord<BaseMap, Result>(base_record, base, result, base_map);
      } else {
        kernels::ExecuteMapRecord<ScaledBase, Result>(
            base_record, base, result, ScaledBase{beta, base_map});
      }
    }
    return;
  }
  if (scalar::IsZero<Beta>(beta)) {
    if (scalar::IsOne<Alpha>(alpha)) {
      kernels::ExecuteMapRecord<SourceMap, Result>(record, source, result, source_map);
    } else {
      kernels::ExecuteMapRecord<ScaledSource, Result>(
          record, source, result, ScaledSource{alpha, source_map});
    }
    return;
  }
  if (scalar::IsOne<Alpha>(alpha)) {
    if (scalar::IsOne<Beta>(beta)) {
      kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
          expression::AddUpdate<SourceMap, BaseMap>{source_map, base_map});
    } else {
      kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
          expression::AddUpdate<SourceMap, ScaledBase>{source_map, {beta, base_map}});
    }
  } else if (scalar::IsOne<Beta>(beta)) {
    kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
        expression::AddUpdate<ScaledSource, BaseMap>{{alpha, source_map}, base_map});
  } else {
    kernels::ExecuteBinaryMapRecord<Result>(record, source, base, result,
        expression::AddUpdate<ScaledSource, ScaledBase>{{alpha, source_map}, {beta, base_map}});
  }
}

// Synchronously execute one complete batch, including output initialization.
template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
void ExecuteUpdateBatch(
    const std::vector<layout::GeneratedRecordProgram>& programs,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t output_size, scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  static_assert(std::is_same_v<typename BaseMap::InputDtype, Result>);
  InitializeUpdateOutput<Result>(base, result, output_size);
  for (const auto& program : programs) {
    layout::ForEachGeneratedBlock(program, [&](const auto& block) {
      ExecuteUpdateRecord<Result, Alpha, Beta>(
          block, source, base, result, alpha, beta, source_map, base_map);
    });
  }
}


// Bind real coefficients in the existing product's real computation type.
// Never turn a real coefficient into a complex one or change half-result products.
template <typename Coefficient, typename Mapped>
using UpdateCoefficient = std::conditional_t<
    !scalar::kIsComplex<Coefficient> &&
        (std::is_same_v<scalar::Component<scalar::Promote<Coefficient, Mapped>>, scalar::F32> ||
         std::is_same_v<scalar::Component<scalar::Promote<Coefficient, Mapped>>, scalar::F64>),
    scalar::Component<scalar::Promote<Coefficient, Mapped>>, Coefficient>;

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
std::function<void(const layout::Record&)> BindUpdateValues(
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<typename BaseMap::InputDtype>* base, scalar::Value<Result>* result,
    scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  return [=](const layout::Record& record) {
    ExecuteUpdateRecord<Result, Alpha, Beta>(record, source, base, result,
        alpha, beta, source_map, base_map);
  };
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
std::function<void(const layout::Record&)> BindUpdateRecord(
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<typename BaseMap::InputDtype>* base, scalar::Value<Result>* result,
    scalar::Value<Alpha> alpha, scalar::Value<Beta> beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  using BoundAlpha = UpdateCoefficient<Alpha, typename SourceMap::OutputDtype>;
  using BoundBeta = UpdateCoefficient<Beta, typename BaseMap::OutputDtype>;
  return BindUpdateValues<Result, BoundAlpha, BoundBeta>(source, base, result,
      scalar::Convert<BoundAlpha, Alpha>(alpha), scalar::Convert<BoundBeta, Beta>(beta),
      source_map, base_map);
}

template <typename Raw, typename Bound>
void ReadUpdateCoefficient(const void* data, uint64_t index, void* destination) {
  *static_cast<scalar::Value<Bound>*>(destination) = scalar::Convert<Bound, Raw>(
      static_cast<const scalar::Value<Raw>*>(data)[index]);
}

template <typename Result>
struct UpdateInit {
  const scalar::Value<Result>* base;
  scalar::Value<Result>* result;
  uint64_t size;
  void operator()(uint64_t batch) const {
    InitializeUpdateOutput<Result>(base + batch * size, result + batch * size, size);
  }
};

template <typename SourceMap, typename BaseMap>
struct UpdateF {
  const scalar::Value<typename SourceMap::InputDtype>* source;
  uint64_t size;
  UpdateCoefficientReader alpha, beta;
  SourceMap source_map;
  BaseMap base_map;
  UpdateF(const scalar::Value<typename SourceMap::InputDtype>* input, uint64_t count,
      UpdateCoefficientReader a, UpdateCoefficientReader b,
      const SourceMap& sm, const BaseMap& bm)
      : source(input), size(count), alpha(a), beta(b), source_map(sm), base_map(bm) {}
};

template <typename Result, typename SourceMap, typename BaseMap>
using UpdateProgram = MapProgram<UpdateF<SourceMap, BaseMap>, UpdateInit<Result>>;

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
const MapProgramEntry& UpdateProgramEntry() {
  using Program = UpdateProgram<Result, SourceMap, BaseMap>;
  static const MapProgramEntry entry{
    [](const void* state, uint64_t batch) { static_cast<const Program*>(state)->init(batch); },
    [](const void* state, uint64_t batch) -> std::function<void(const layout::Record&)> {
      const auto& p = *static_cast<const Program*>(state);
      scalar::Value<Alpha> alpha;
      scalar::Value<Beta> beta;
      p.f.alpha.load(p.f.alpha.data, p.f.alpha.count == 1 ? 0 : batch, &alpha);
      p.f.beta.load(p.f.beta.data, p.f.beta.count == 1 ? 0 : batch, &beta);
      return BindUpdateValues<Result, Alpha, Beta>(
          p.f.source == nullptr ? nullptr : p.f.source + batch * p.f.size,
          p.init.base + batch * p.init.size, p.init.result + batch * p.init.size,
          alpha, beta, p.f.source_map, p.f.base_map);
    }
  };
  return entry;
}

template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
ffi::Future ExecuteBoundUpdate(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    UpdateCoefficientReader alpha, UpdateCoefficientReader beta,
    const SourceMap& source_map, const BaseMap& base_map) {
  static_assert(std::is_same_v<typename BaseMap::InputDtype, Result>);
  try {
    auto state = MakeProgramState<UpdateProgram<Result, SourceMap, BaseMap>>(
        UpdateInit<Result>{base, result, output_size}, source, source_size, alpha, beta,
        source_map, base_map);
    if ((alpha.count != 1 && alpha.count != batch_count) ||
        (beta.count != 1 && beta.count != batch_count)) {
      throw std::invalid_argument("update coefficients must be shared or one value per batch");
    }
    return ExecuteOwnedMapProgram(thread_pool, records, sizeof(*source), sizeof(*result),
        output_size, batch_count,
        {std::move(state), &UpdateProgramEntry<Result, Alpha, Beta, SourceMap, BaseMap>()});
  } catch (const std::exception& error) {
    return CompletedFuture(ffi::Error::Internal(std::string("tensor0-native: ") + error.what()));
  } catch (...) {
    return CompletedFuture(ffi::Error::Internal("tensor0-native: unknown map preparation exception"));
  }
}

// Typed callers select independent readers; only bound types reach the executor.
template <typename Result, typename Alpha, typename Beta, typename SourceMap, typename BaseMap>
ffi::Future ExecuteUpdate(
    ffi::ThreadPool thread_pool, const std::vector<layout::Record>& records,
    const scalar::Value<typename SourceMap::InputDtype>* source,
    const scalar::Value<Result>* base, scalar::Value<Result>* result,
    uint64_t source_size, uint64_t output_size, uint64_t batch_count,
    const scalar::Value<Alpha>* alpha, uint64_t alpha_count,
    const scalar::Value<Beta>* beta, uint64_t beta_count,
    SourceMap source_map, BaseMap base_map) {
  using BoundAlpha = UpdateCoefficient<Alpha, typename SourceMap::OutputDtype>;
  using BoundBeta = UpdateCoefficient<Beta, typename BaseMap::OutputDtype>;
  return ExecuteBoundUpdate<Result, BoundAlpha, BoundBeta>(thread_pool, records,
      source, base, result, source_size, output_size, batch_count,
      {alpha, alpha_count, ReadUpdateCoefficient<Alpha, BoundAlpha>},
      {beta, beta_count, ReadUpdateCoefficient<Beta, BoundBeta>},
      source_map, base_map);
}

}
