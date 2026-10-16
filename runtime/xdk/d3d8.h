#pragma once
#include <mutex>

#include "gpu/state.h"

namespace d3d {

gpu::State& State();
std::recursive_mutex& Lock();

// A title's say over a pixel shader constant it sets: called with the call's return address, the register and its
// four values, which it may change. Null: none.
using PixelConstantFilter = void (*)(uint32_t from, uint32_t reg, float* value);
void SetPixelConstantFilter(PixelConstantFilter filter);

// A title's own presentation interval for the swaps that follow (vblanks, 1 to 4), whatever it set on the device:
// one whose logic steps at 60 Hz can show each step. 0: the title's.
void SetIntervalOverride(uint32_t vblanks);

}  // namespace d3d
