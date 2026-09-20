#pragma once

#include "scalar.h"

namespace tensor0::stride::expression {

template <typename Dtype>
struct Identity {
  using InputDtype = Dtype;
  using OutputDtype = Dtype;

  scalar::Value<Dtype> operator()(scalar::Value<Dtype> value) const {
    return value;
  }
};

template <typename Source, typename Result = Source>
struct Zero {
  using InputDtype = Source;
  using OutputDtype = Result;

  scalar::Value<Result> operator()(scalar::Value<Source>) const {
    return {};
  }
};

template <typename Result, typename Map = Identity<Result>>
struct Cast {
  using InputDtype = typename Map::InputDtype;
  using OutputDtype = Result;

  Map map{};

  scalar::Value<Result> operator()(scalar::Value<InputDtype> value) const {
    return scalar::Convert<Result, typename Map::OutputDtype>(map(value));
  }
};

template <typename Dtype>
struct Conjugate {
  using InputDtype = Dtype;
  using OutputDtype = Dtype;

  scalar::Value<Dtype> operator()(scalar::Value<Dtype> value) const {
    return scalar::Conjugate<Dtype>(value);
  }
};

template <typename Coefficient, typename Map = Identity<Coefficient>>
struct Scale {
  using InputDtype = typename Map::InputDtype;
  using OutputDtype = scalar::Promote<Coefficient, typename Map::OutputDtype>;

  scalar::Value<Coefficient> factor;
  Map map{};

  scalar::Value<OutputDtype> operator()(scalar::Value<InputDtype> value) const {
    return scalar::Multiply<Coefficient, typename Map::OutputDtype>(factor, map(value));
  }
};

template <typename SourceOp, typename BaseOp>
struct AddUpdate {
  using InputDtype = typename SourceOp::InputDtype;
  using BaseInputDtype = typename BaseOp::InputDtype;
  using OutputDtype = scalar::Promote<typename SourceOp::OutputDtype,
                                      typename BaseOp::OutputDtype>;

  SourceOp source_op{};
  BaseOp base_op{};

  scalar::Value<OutputDtype> operator()(
      scalar::Value<InputDtype> source,
      scalar::Value<BaseInputDtype> original_base) const {
    const auto source_term = source_op(source);
    const auto base_term = base_op(original_base);
    return scalar::Add<typename SourceOp::OutputDtype, typename BaseOp::OutputDtype>(
        source_term, base_term);
  }
};

}
