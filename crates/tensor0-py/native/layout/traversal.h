#pragma once

#include "record.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor0::stride::layout {

std::vector<uint64_t> GeneratedIndexOrder(
    const std::vector<int64_t>& strides);

std::vector<std::size_t> ComputeLocalityOrder(const Record& record);

void SortRecordDimensions(Record* record);

void OptimizeRecordForExecution(Record* record);

bool IsCompactSameMapping(const Record& record);

bool IsRank2ForwardCandidate(const Record& record);

bool IsRank2ReverseCandidate(const Record& record);

uint64_t GenericRowCount(const Record& record);

bool IsBroadcastContiguousRowCandidate(const Record& record);

bool IsContiguousInnerRowsCandidate(const Record& record);

bool IsRank4TwoPairCandidate(const Record& record, std::array<std::size_t, 4>* axes);

}
