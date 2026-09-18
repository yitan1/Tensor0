#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <sys/mman.h>
#include <unistd.h>
#include "numeric/scalar.inc"
#include "numeric/expression.inc"
#include "layout/types.inc"
#include "layout/address.inc"
#include "layout/construction.inc"
#include "layout/planning.inc"
#include "layout/blocking.inc"
#include "kernels/affine.inc"
#include "execute/update.inc"

void CheckBranches() {
  auto record = layout::BuildLayout({2}, {-2}, 5, {-2}, 5, 8, 8, 0);
  auto scalar_record = layout::BuildLayout({}, {}, 1, {}, 1, 8, 8, 1);
  layout::OptimizeRecordForExecution(&record);
  const std::vector<layout::Record> records{record, scalar_record};
  std::array<float, 24> storage;
  for (std::size_t index = 0; index < storage.size(); ++index) storage[index] = index;
  const auto original = storage;
  const std::array<double, 3> factors{0, 1, 2};
  const int32_t zero = 0;
  ExecuteUpdateBatches<scalar::F32, scalar::F64, scalar::S32>(
      records, storage.data(), storage.data(), storage.data(), 8, 8, 3, factors.data(), 3,
      &zero, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  for (std::size_t batch = 0; batch < 3; ++batch) {
    for (std::size_t index = 0; index < 8; ++index) {
      const bool selected = index == 1 || index == 3 || index == 5;
      assert(storage[batch * 8 + index] == original[batch * 8 + index] * (selected ? factors[batch] : 1));
    }
  }
  const auto before = storage;
  bool rejected = false;
  try {
    ExecuteUpdateBatches<scalar::F32, scalar::F64, scalar::S32>(
        records, storage.data(), storage.data(), storage.data(), 8, 8, 3, factors.data(), 2,
        &zero, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && storage == before);
  ExecuteUpdateBatches<scalar::F32, scalar::F64, scalar::S32>(
      records, static_cast<const float*>(nullptr), nullptr, nullptr, 8, 8, 0, nullptr, 0,
      &zero, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
  ExecuteUpdateBatches<scalar::F32, scalar::F64, scalar::S32>(
      {}, static_cast<const float*>(nullptr), nullptr, nullptr, 0, 0, 2, nullptr, 1,
      &zero, 1, expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
}

void CheckNoWholeStorageAccess() {
  const auto page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  auto* mapping = mmap(nullptr, 3 * page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(mapping != MAP_FAILED);
  auto* middle = static_cast<char*>(mapping) + page_size;
  assert(mprotect(middle, page_size, PROT_READ | PROT_WRITE) == 0);
  auto* storage = static_cast<float*>(mapping);
  const int64_t offset = page_size / sizeof(float);
  storage[offset] = 3;
  const uint64_t size = 3 * page_size / sizeof(float);
  const auto record = layout::BuildLayout({}, {}, offset, {}, offset, size, size, 0);
  for (float factor : {2.0F, 1.0F, 0.0F}) {
    const auto expected = storage[offset] * factor;
    ExecuteUpdate<scalar::F32, scalar::F32, scalar::S32>(
        {record}, storage, storage, storage, size, factor, 0,
        expression::Identity<scalar::F32>{}, expression::Identity<scalar::F32>{});
    assert(storage[offset] == expected);
  }
  assert(munmap(mapping, 3 * page_size) == 0);
}

int main() {
  CheckBranches();
  CheckNoWholeStorageAccess();
}
