#pragma once

#include "xla/ffi/api/ffi.h"
#include <exception>
#include <stdexcept>
#include <string>

namespace tensor0::stride {

namespace ffi = xla::ffi;

template <typename Execute>
ffi::Error ContainErrors(Execute execute) {
  try {
    execute();
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(std::string("tensor0-native: ") + error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(std::string("tensor0-native: ") + error.what());
  } catch (...) {
    return ffi::Error::Internal("tensor0-native: unknown exception");
  }
}

}
