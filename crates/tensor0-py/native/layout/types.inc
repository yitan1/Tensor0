#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace layout {

struct Record {
  std::size_t semantic_index = 0;
  int64_t source_offset = 0;
  int64_t destination_offset = 0;
  std::vector<uint64_t> shape;
  std::vector<int64_t> source_strides;
  std::vector<int64_t> destination_strides;
};


}
