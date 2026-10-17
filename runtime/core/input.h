#pragma once
#include <array>
#include <cstdint>

namespace input {

// the console controller's report, analog buttons are 0-255
struct Pad {
    bool connected = false;
    uint16_t buttons = 0;  // dpad up/down/left/right, start, back, left thumb, right thumb
    uint8_t analog[8] = {};  // A B X Y black white left trigger right trigger
    int16_t lx = 0, ly = 0, rx = 0, ry = 0;
    bool keyboard = false;  // keys held went into it (an overlay with its own key handling can tell)
};

// The keyboard works controller 1: each of its controls by up to two keys (Windows virtual-key codes, 0 none),
// kept in the settings file's [keys]
enum class Control {
    kStickLeft, kStickRight, kStickUp, kStickDown, kRightTrigger, kLeftTrigger,
    kA, kB, kX, kY, kBlack, kWhite, kStart, kBack,
    kDpadUp, kDpadDown, kDpadLeft, kDpadRight,
    kRightStickUp, kRightStickDown, kRightStickLeft, kRightStickRight, kLeftThumb, kRightThumb,
    kCount
};
constexpr int kControls = int(Control::kCount);
using KeyMap = std::array<std::array<uint16_t, 2>, kControls>;
KeyMap& Keys();
const KeyMap& DefaultKeys();
const char* ControlId(int control);  // its name in [keys]
const char* KeyName(uint16_t vk);    // in capitals, as on a US keyboard ("" for none)

void Init();  // window thread, before boot
// the controller on that port, then GameInfo::pad
Pad Poll(int port);
void Rumble(int port, uint16_t left, uint16_t right);

}  // namespace input
