#pragma once

#include "record.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensor0::stride::layout {

void OptimizeRecordForExecution(Record* record);

bool IsCompactSameMapping(const Record& record);

bool IsRank2ForwardCandidate(const Record& record);

bool IsRank2ReverseCandidate(const Record& record);

uint64_t GenericRowCount(const Record& record);

bool IsBroadcastContiguousRowCandidate(const Record& record);

bool IsContiguousInnerRowsCandidate(const Record& record);

bool IsRank4TwoPairCandidate(const Record& record, std::array<std::size_t, 4>* axes);

}
