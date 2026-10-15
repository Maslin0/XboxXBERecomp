#pragma once
#include <cstdint>
#include <functional>

namespace audio {

constexpr uint32_t kRate = 48000;

// the platform output pulls interleaved stereo float frames from render
void StartOutput(std::function<void(float* out, uint32_t frames)> render);
// paused: the output plays silence and render is not called, so every voice stays where it is
void SetPaused(bool paused);

}  // namespace audio
