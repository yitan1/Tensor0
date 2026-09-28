#pragma once

#include "thread_pool_test_support.h"

namespace testing {

template <ffi::DataType Dtype, typename... Args>
ffi::Error Update(Args&&... args) {
  ThreadPool pool;
  return CompletedError(tensor0::stride::Update<Dtype>(std::forward<Args>(args)..., pool.get()));
}

}
