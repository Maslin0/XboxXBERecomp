// controllers through SDL: Xbox, PlayStation and Switch pads read into the console controller's report
#define NOMINMAX
#include "core/input.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <mutex>

#include <SDL3/SDL.h>
#include <windows.h>

#include "app/app.h"
#include "core/log.h"

namespace input {

namespace {

// a controller takes the lowest free port when it is plugged in and keeps it until it is unplugged
std::mutex g_lock;
bool g_sdl = false;
SDL_Gamepad* g_pads[4] = {};

void AddPad(SDL_JoystickID id) {
    for (SDL_Gamepad* pad : g_pads)
        if (pad && SDL_GetGamepadID(pad) == id) return;
    SDL_Gamepad** slot = std::find(std::begin(g_pads), std::end(g_pads), nullptr);
    if (slot == std::end(g_pads)) return;
    *slot = SDL_OpenGamepad(id);
    const char* name = *slot ? SDL_GetGamepadName(*slot) : SDL_GetError();
    LOGI("input: controller %d %s: %s", int(slot - g_pads) + 1, *slot ? "plugged in" : "failed to open", name ? name : "");
}

void RemovePad(SDL_JoystickID id) {
    for (int i = 0; i < 4; i++) {
        if (!g_pads[i] || SDL_GetGamepadID(g_pads[i]) != id) continue;
        SDL_CloseGamepad(g_pads[i]);
        g_pads[i] = nullptr;
        LOGI("input: controller %d unplugged", i + 1);
    }
}

// once a millisecond at most, however many ports and threads read
void UpdatePads() {
    static uint64_t last = 0;
    uint64_t now = SDL_GetTicks();
    if (!g_sdl || now == last) return;
    last = now;
    SDL_PumpEvents();
    SDL_Event e;
    while (SDL_PeepEvents(&e, 1, SDL_GETEVENT, SDL_EVENT_FIRST, SDL_EVENT_LAST) == 1) {
        if (e.type == SDL_EVENT_GAMEPAD_ADDED) AddPad(e.gdevice.which);
        else if (e.type == SDL_EVENT_GAMEPAD_REMOVED) RemovePad(e.gdevice.which);
    }
}

// The keyboard as controller 1, while the game window has the focus, by Keys(). By default:
//   steer A D / Left Right (left stick), throttle W / Up (right trigger), brake S / Down (left trigger); W S and the
//   arrows also move the left stick up and down for the menus
//   A Enter Space, B Backspace, X Left Shift, Y Left Ctrl, black Q, white E, Start Esc, Back Tab,
//   right stick I J K L, d-pad the numpad's 8 2 4 6, thumbs F (left) R (right)
// In the title's menus (SetMenuKeys) the arrows are the d-pad, Enter and Space A, Backspace and Escape B, whatever
// they are bound to, and work nothing else
constexpr int kMenuKeys[] = {VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN, VK_SPACE, VK_BACK, VK_ESCAPE};
std::atomic<bool> g_menu_keys{false};
std::atomic<bool> g_keyboard_last{false};  // the keyboard used since controller 1 last was

bool Down(int vk) { return GetAsyncKeyState(vk) & 0x8000; }

void Keyboard(Pad& p) {
    if (!app::Get().keyboard) return;
    p.connected = true;  // a controller is there for the game even with none plugged in
    HWND window = static_cast<HWND>(app::Window());
    if (!window || GetForegroundWindow() != window) return;
    const KeyMap& keys = Keys();
    bool menu = g_menu_keys;
    auto fixed = [&](uint16_t vk) {
        return menu && std::find(std::begin(kMenuKeys), std::end(kMenuKeys), vk) != std::end(kMenuKeys);
    };
    auto held = [&](Control c) {
        for (uint16_t vk : keys[size_t(c)])
            if (vk && !fixed(vk) && Down(vk)) return true;
        return false;
    };
    bool any = false;
    if (menu) {
        static const uint16_t kDpad[] = {VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT};
        for (int i = 0; i < 4; i++)
            if (Down(kDpad[i])) p.buttons |= uint16_t(1u << i), any = true;
        if (Down(VK_RETURN) || Down(VK_SPACE)) p.analog[0] = 0xFF, any = true;
        if (Down(VK_BACK) || Down(VK_ESCAPE)) p.analog[1] = 0xFF, any = true;
    }
    auto button = [&](uint16_t bit, Control c) {
        if (held(c)) p.buttons |= bit, any = true;
    };
    auto analog = [&](int i, Control c) {
        if (held(c)) p.analog[i] = 0xFF, any = true;
    };
    auto axis = [&](int16_t& a, Control minus, Control plus) {
        int v = (held(plus) ? 32767 : 0) - (held(minus) ? 32767 : 0);
        if (v) a = int16_t(v), any = true;
    };
    axis(p.lx, Control::kStickLeft, Control::kStickRight);
    axis(p.ly, Control::kStickDown, Control::kStickUp);
    axis(p.rx, Control::kRightStickLeft, Control::kRightStickRight);
    axis(p.ry, Control::kRightStickDown, Control::kRightStickUp);
    analog(7, Control::kRightTrigger);
    analog(6, Control::kLeftTrigger);
    analog(0, Control::kA);
    analog(1, Control::kB);
    analog(2, Control::kX);
    analog(3, Control::kY);
    analog(4, Control::kBlack);
    analog(5, Control::kWhite);
    button(0x01, Control::kDpadUp);
    button(0x02, Control::kDpadDown);
    button(0x04, Control::kDpadLeft);
    button(0x08, Control::kDpadRight);
    button(0x10, Control::kStart);
    button(0x20, Control::kBack);
    button(0x40, Control::kLeftThumb);
    button(0x80, Control::kRightThumb);
    p.keyboard = any;
}

}  // namespace

const KeyMap& DefaultKeys() {
    static const KeyMap keys = {{
        {'A', VK_LEFT}, {'D', VK_RIGHT}, {'W', VK_UP}, {'S', VK_DOWN}, {'W', VK_UP}, {'S', VK_DOWN},
        {VK_RETURN, VK_SPACE}, {VK_BACK, 0}, {VK_LSHIFT, 0}, {VK_LCONTROL, 0}, {'Q', 0}, {'E', 0},
        {VK_ESCAPE, 0}, {VK_TAB, 0},
        {VK_NUMPAD8, 0}, {VK_NUMPAD2, 0}, {VK_NUMPAD4, 0}, {VK_NUMPAD6, 0},
        {'I', 0}, {'K', 0}, {'J', 0}, {'L', 0}, {'F', 0}, {'R', 0},
    }};
    return keys;
}

KeyMap& Keys() {
    static KeyMap keys = DefaultKeys();
    return keys;
}

const char* ControlId(int control) {
    static const char* ids[kControls] = {
        "stick_left", "stick_right", "stick_up", "stick_down", "right_trigger", "left_trigger",
        "a", "b", "x", "y", "black", "white", "start", "back",
        "dpad_up", "dpad_down", "dpad_left", "dpad_right",
        "right_stick_up", "right_stick_down", "right_stick_left", "right_stick_right", "left_thumb", "right_thumb",
    };
    return control >= 0 && control < kControls ? ids[control] : "";
}

const char* KeyName(uint16_t vk) {
    static char buf[16];
    if (!vk) return "";
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) {
        buf[0] = char(vk);
        buf[1] = 0;
        return buf;
    }
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) {
        std::snprintf(buf, sizeof(buf), "NUM %d", vk - VK_NUMPAD0);
        return buf;
    }
    if (vk >= VK_F1 && vk <= VK_F24) {
        std::snprintf(buf, sizeof(buf), "F%d", vk - VK_F1 + 1);
        return buf;
    }
    switch (vk) {
        case VK_RETURN: return "ENTER";
        case VK_SPACE: return "SPACE";
        case VK_BACK: return "BACKSPACE";
        case VK_TAB: return "TAB";
        case VK_ESCAPE: return "ESC";
        case VK_LSHIFT: return "L SHIFT";
        case VK_RSHIFT: return "R SHIFT";
        case VK_LCONTROL: return "L CTRL";
        case VK_RCONTROL: return "R CTRL";
        case VK_LMENU: return "L ALT";
        case VK_RMENU: return "R ALT";
        case VK_UP: return "UP";
        case VK_DOWN: return "DOWN";
        case VK_LEFT: return "LEFT";
        case VK_RIGHT: return "RIGHT";
        case VK_INSERT: return "INSERT";
        case VK_DELETE: return "DELETE";
        case VK_HOME: return "HOME";
        case VK_END: return "END";
        case VK_PRIOR: return "PAGE UP";
        case VK_NEXT: return "PAGE DOWN";
        case VK_CAPITAL: return "CAPS LOCK";
        case VK_MULTIPLY: return "NUM *";
        case VK_ADD: return "NUM +";
        case VK_SUBTRACT: return "NUM -";
        case VK_DECIMAL: return "NUM .";
        case VK_DIVIDE: return "NUM /";
        case VK_OEM_1: return ";";
        case VK_OEM_PLUS: return "=";
        case VK_OEM_COMMA: return ",";
        case VK_OEM_MINUS: return "-";
        case VK_OEM_PERIOD: return ".";
        case VK_OEM_2: return "/";
        case VK_OEM_3: return "`";
        case VK_OEM_4: return "[";
        case VK_OEM_5: return "\\";
        case VK_OEM_6: return "]";
        case VK_OEM_7: return "'";
        default:
            std::snprintf(buf, sizeof(buf), "KEY %d", vk);
            return buf;
    }
}

void SetMenuKeys(bool menu) { g_menu_keys = menu; }

void Init() {
    // WGI reads the triggers apart and HIDAPI takes PlayStation and Switch pads. Reads come from the guest's threads,
    // so SDL's own thread takes the device notices.
    SDL_SetHint(SDL_HINT_JOYSTICK_THREAD, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_WGI, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT, "0");
    SDL_SetHint(SDL_HINT_XINPUT_ENABLED, "0");
    SDL_SetHint(SDL_HINT_JOYSTICK_DIRECTINPUT, "0");
    if (!SDL_Init(SDL_INIT_GAMEPAD)) {
        LOGE("input: SDL failed to start, no controllers: %s", SDL_GetError());
        return;
    }
    int v = SDL_GetVersion();
    LOGI("input: SDL %d.%d.%d", SDL_VERSIONNUM_MAJOR(v), SDL_VERSIONNUM_MINOR(v), SDL_VERSIONNUM_MICRO(v));
    std::lock_guard lock(g_lock);
    g_sdl = true;
    int count = 0;
    if (SDL_JoystickID* ids = SDL_GetGamepads(&count)) {
        for (int i = 0; i < count; i++) AddPad(ids[i]);
        SDL_free(ids);
    }
}

Pad Poll(int port) {
    Pad p;
    {
        std::lock_guard lock(g_lock);
        UpdatePads();
        if (SDL_Gamepad* pad = port >= 0 && port < 4 ? g_pads[port] : nullptr) {
            p.connected = true;
            constexpr SDL_GamepadButton kButtons[] = {
                SDL_GAMEPAD_BUTTON_DPAD_UP,   SDL_GAMEPAD_BUTTON_DPAD_DOWN,  SDL_GAMEPAD_BUTTON_DPAD_LEFT,
                SDL_GAMEPAD_BUTTON_DPAD_RIGHT, SDL_GAMEPAD_BUTTON_START,     SDL_GAMEPAD_BUTTON_BACK,
                SDL_GAMEPAD_BUTTON_LEFT_STICK, SDL_GAMEPAD_BUTTON_RIGHT_STICK};
            for (int i = 0; i < 8; i++) p.buttons |= uint16_t(SDL_GetGamepadButton(pad, kButtons[i]) << i);
            // A B X Y black white
            constexpr SDL_GamepadButton kAnalog[] = {SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_EAST,
                                                     SDL_GAMEPAD_BUTTON_WEST,  SDL_GAMEPAD_BUTTON_NORTH,
                                                     SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER};
            for (int i = 0; i < 6; i++) p.analog[i] = SDL_GetGamepadButton(pad, kAnalog[i]) ? 0xFF : 0;
            auto axis = [&](SDL_GamepadAxis a) { return int(SDL_GetGamepadAxis(pad, a)); };
            // 0 to 32767 into 0 to 255
            p.analog[6] = uint8_t(std::max(axis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER), 0) >> 7);
            p.analog[7] = uint8_t(std::max(axis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER), 0) >> 7);
            // SDL's sticks have y down
            // not plain negation: -32768 has no positive
            p.lx = int16_t(axis(SDL_GAMEPAD_AXIS_LEFTX));
            p.ly = int16_t(-1 - axis(SDL_GAMEPAD_AXIS_LEFTY));
            p.rx = int16_t(axis(SDL_GAMEPAD_AXIS_RIGHTX));
            p.ry = int16_t(-1 - axis(SDL_GAMEPAD_AXIS_RIGHTY));
        }
    }
    if (port == 0) {
        // whichever was used last, the keyboard or controller 1, for a title's prompts
        constexpr int16_t kPushed = 16000;
        auto pushed = [](int16_t v) { return v > kPushed || v < -kPushed; };
        bool pad = p.buttons || pushed(p.lx) || pushed(p.ly) || pushed(p.rx) || pushed(p.ry);
        for (uint8_t a : p.analog) pad |= a > 0x20;
        if (pad) g_keyboard_last = false;
        Keyboard(p);
        if (p.keyboard) g_keyboard_last = true;
    }
    if (app::kGame.pad) app::kGame.pad(port, p);
    return p;
}

bool KeyboardActive() { return g_keyboard_last && app::Get().keyboard; }

// the motors run until set again
void Rumble(int port, uint16_t left, uint16_t right) {
    if (app::kGame.rumble) app::kGame.rumble(port, left, right);
    std::lock_guard lock(g_lock);
    if (port >= 0 && port < 4 && g_pads[port]) SDL_RumbleGamepad(g_pads[port], left, right, 0);
}

}  // namespace input
