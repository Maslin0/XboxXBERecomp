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

}  // namespace d3d
