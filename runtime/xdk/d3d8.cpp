// Direct3D 8 (XDK 5849) entry points the title calls, implemented on the host renderer.
// Resource creation, locking and the XG helpers stay recompiled: they only manage memory.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "app/app.h"
#include "core/diag.h"
#include "gpu/backend.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "xdk/d3d8.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

using namespace kernel;

// configs written before the SDK knew the name
#ifndef SYM_D3DDevice_BeginPush
#define SYM_D3DDevice_BeginPush FFFFFE00
#define HAS_D3DDevice_BeginPush 0
#endif
#ifndef SYM_D3DDevice_EndPush
#define SYM_D3DDevice_EndPush FFFFFE01
#define HAS_D3DDevice_EndPush 0
#endif
#ifndef SYM_D3DDevice_RunPushBuffer
#define SYM_D3DDevice_RunPushBuffer FFFFFE05
#define HAS_D3DDevice_RunPushBuffer 0
#endif
#ifndef SYM_D3DDevice_GetVisibilityTestResult
#define SYM_D3DDevice_GetVisibilityTestResult FFFFFE04
#define HAS_D3DDevice_GetVisibilityTestResult 0
#endif
// CDevice's table of visibility report blocks, 256 reports of 16 bytes each (XDK 5849)
#ifndef XDK_CDevice_VisibilityReports
#define XDK_CDevice_VisibilityReports 0x7D4u
#endif
#ifndef HAS_D3D_FvfShader
#define HAS_D3D_FvfShader 0
#endif
#ifndef SYM_D3DDevice_UpdateOverlay
#define SYM_D3DDevice_UpdateOverlay FFFFFE06
#define HAS_D3DDevice_UpdateOverlay 0
#endif
#ifndef SYM_D3DDevice_GetOverlayUpdateStatus
#define SYM_D3DDevice_GetOverlayUpdateStatus FFFFFE07
#define HAS_D3DDevice_GetOverlayUpdateStatus 0
#endif
#ifndef SYM_D3DDevice_SetPalette
#define SYM_D3DDevice_SetPalette FFFFFE02
#define HAS_D3DDevice_SetPalette 0
#endif
#ifndef SYM_D3DDevice_GetVertexShaderFunction
#define SYM_D3DDevice_GetVertexShaderFunction FFFFFE03
#define HAS_D3DDevice_GetVertexShaderFunction 0
#endif

namespace d3d {

namespace {

// A/B switch: 0 draws on the title's thread, as before
#ifndef XBE_RENDER_THREAD
#define XBE_RENDER_THREAD 1
#endif
constexpr bool kRenderThread = XBE_RENDER_THREAD;

// D3D globals of this build
constexpr uint32_t kDevice = SYM(D3D_Device);        // the static CDevice, g_pDevice points at it
constexpr uint32_t kDevicePtr = SYM(D3D_pDevice);
constexpr uint32_t kIndexData = SYM(D3D_IndexData);  // index data base the inline draws read
constexpr uint32_t kDirtyFlags = SYM(D3D_DirtyFlags);
constexpr uint32_t kTextureState = SYM(D3D_DeferredTextureState);  // [4][32]
constexpr uint32_t kRenderState = SYM(D3D_RenderState);            // [166]
constexpr uint32_t kStreams = SYM(D3D_Streams);                    // [16] stride, offset, buffer
constexpr uint32_t kSimpleEncode = SYM(D3D_SimpleRenderStateEncode);

// CDevice fields mirrored so the recompiled getters and resource code see what the XDK would
constexpr uint32_t kDevPush = XDK_CDevice_Push, kDevThreshold = XDK_CDevice_PushThreshold,
                   kDevBaseVertex = XDK_CDevice_BaseVertex, kDevIndexBuffer = XDK_CDevice_IndexBuffer,
                   kDevTextures = XDK_CDevice_Textures, kDevRenderTarget = XDK_CDevice_RenderTarget,
                   kDevDepth = XDK_CDevice_Depth, kDevSwapCount = XDK_CDevice_SwapCount,
                   kDevBackBuffer = XDK_CDevice_BackBuffer, kDevFrontBuffer = XDK_CDevice_FrontBuffer,
                   kDevZScale = XDK_CDevice_ZScale, kDevSuperX = XDK_CDevice_SuperSampleX,
                   kDevSuperY = XDK_CDevice_SuperSampleY, kDevTransforms = XDK_CDevice_Transforms,
                   kDevPalettes = XDK_CDevice_Palettes;
constexpr uint32_t kStageStates = XDK_TextureState_Stage, kStreamSize = XDK_Stream_Size;

gpu::State g_state;

// Hardware vertex state a title pushes itself, past the API: some engines load vertex programs and constants and
// point the vertex arrays with their own commands, then run recorded index lists (RunPushBuffer). Those draws take
// their shader and streams from here.
struct RawVertex {
    std::array<uint32_t, 136 * 4> program{};  // transform program memory, 136 instructions
    uint32_t load = 0, start = 0;             // SET_TRANSFORM_PROGRAM_LOAD, _START, in instructions
    uint32_t load_word = 0;                   // dword within the instruction being loaded
    uint32_t const_load = 0, const_word = 0;  // SET_TRANSFORM_CONSTANT_LOAD, hardware register order
    std::array<uint32_t, 16> offset{}, format{};  // SET_VERTEX_DATA_ARRAY_OFFSET, _FORMAT
    bool arrays = false;                      // the title has pointed the arrays itself
    uint32_t selected = 0;                    // the last handle selected or set: its declaration lays out the streams
    bool own_program = false;                 // the program runs from the title's program memory (selected or pushed)
    std::array<uint32_t, 16> headers{};       // vertex buffer headers the streams point at
};
RawVertex g_raw;
uint32_t g_interval = 0;  // D3DPRESENT_PARAMETERS presentation interval
void PushStat(uint32_t m);
gpu::State RawDrawState();
gpu::State SelectedProgramState();
bool DirectHandle(uint32_t h);
gpu::State DrawState();
std::unique_ptr<gpu::Backend> g_backend;
std::recursive_mutex g_lock;

std::unordered_map<uint32_t, std::shared_ptr<gpu::VertexShader>> g_vs;
std::unordered_map<uint32_t, std::shared_ptr<gpu::PixelShader>> g_ps;
std::array<uint32_t, 3> g_buffers{};  // front, back surfaces
uint32_t g_buffer_count = 2;
uint32_t g_depth = 0;
uint32_t g_scratch_push = 0, g_scratch_size = 0;
uint32_t g_walk = 0;  // first scratch command not read yet

uint32_t Dev(uint32_t off) { return R32(kDevice + off); }
void SetDev(uint32_t off, uint32_t v) { W32(kDevice + off, v); }

// The device's own references: a count in bits 19 to 22 of the resource's first dword. A surface's first one
// also holds its parent. The library's functions do it when the image has them, inlined ones are done here.
void AddRef(X86Context& ctx, uint32_t res) {
    if (!res) return;
#if HAS_D3D_InternalAddRef
    GUEST_CALL(D3D_InternalAddRef, ctx, {res});
#else
    (void)ctx;
    uint32_t common = R32(res);
    if (!(common & 0x780000) && (common & 0x70000) == 0x50000)
        if (uint32_t parent = R32(res + 0x14)) W32(parent, R32(parent) + 0x80000);
    W32(res, R32(res) + 0x80000);
#endif
}
void Release(X86Context& ctx, uint32_t res) {
    if (!res) return;
#if HAS_D3D_InternalRelease
    GUEST_CALL(D3D_InternalRelease, ctx, {res});
#else
    (void)ctx;
    // the library frees a resource nothing holds any more; without its function the memory stays
    auto drop = [](uint32_t r) {
        W32(r, R32(r) - 0x80000);
        if (!(R32(r) & 0x78FFFF)) LOG_FIRST(8, "d3d: resource %08X is not freed, the image has no D3D_InternalRelease", r);
    };
    uint32_t common = R32(res);
    if ((common & 0x780000) == 0x80000 && (common & 0x70000) == 0x50000)
        if (uint32_t parent = R32(res + 0x14)) drop(parent);
    drop(res);
#endif
}

void Walk(uint32_t end);

// Hardware commands a caller still pushes go to a scratch buffer, rewound at every request once Walk has read it
uint32_t ScratchPush(uint32_t bytes = 0) {
    Walk(Dev(kDevPush));
    uint32_t need = std::max<uint32_t>(0x10000, (bytes + 0x1FFF) & ~0xFFFu);
    if (need > g_scratch_size) {
        if (g_scratch_push) mem::FreeContiguous(g_scratch_push);
        g_scratch_push = mem::AllocContiguous(need, 0, 0xFFFFFFFF, 0x1000);
        g_scratch_size = need;
    }
    SetDev(kDevPush, g_scratch_push);
    SetDev(kDevThreshold, g_scratch_push + g_scratch_size - 0x1000);
    g_walk = g_scratch_push;
    return g_scratch_push;
}

gpu::Surface SurfaceOf(uint32_t header) {
    gpu::Surface s;
    s.header = header;
    if (header) {
        s.data = R32(header + 4);
        s.format = R32(header + 12);
        s.size = R32(header + 16);
    }
    return s;
}

uint32_t NewSurface(uint32_t w, uint32_t h, uint32_t fmt, uint32_t bpp) {
    uint32_t pitch = (w * bpp + 63) & ~63u;
    uint32_t data = mem::AllocContiguous(pitch * h, 0, 0xFFFFFFFF, 0x1000);
    uint32_t hdr = mem::AllocSystem(24);
    W32(hdr + 0, 0x000D0001);  // surface, one reference, one internal reference held by the device
    W32(hdr + 4, mem::PhysicalFromVirtual(data));
    W32(hdr + 8, 0);
    W32(hdr + 12, 0x00010021 | (fmt << 8));  // dma A, 2D, one level
    W32(hdr + 16, (w - 1) | (h - 1) << 12 | (pitch / 64 - 1) << 24);
    W32(hdr + 20, 0);
    return hdr;
}

void SyncDeferred() {
    // deferred states live in the title's memory, the inline SetRenderState and SetTextureStageState write them
    for (uint32_t i = gpu::kRsDeferredFirst; i < gpu::kRsComplexFirst; i++) g_state.rs[i] = R32(kRenderState + i * 4);
    // titles that write the library's table themselves change the texture modes past SetRenderState
    g_state.rs[gpu::kRsPsTextureModes] = R32(kRenderState + gpu::kRsPsTextureModes * 4);
    for (int s = 0; s < 4; s++)
        for (uint32_t i = 0; i < gpu::kTssBumpEnvMat00; i++) g_state.tss[s][i] = R32(kTextureState + (s * kStageStates + i) * 4);
}

float ZScale(const gpu::Surface& zs) {
    uint32_t f = zs.header ? (zs.format >> 8) & 0xFF : 0x2E;
    return f == 0x2C || f == 0x30 ? 65535.0f : 16777215.0f;
}

bool g_own_constants = false;  // D3DSCM_NORESERVEDCONSTANTS: c-38 and c-37 are the title's

// c-38 and c-37 hold the viewport scale and offset, as D3D_GetViewportOffsetAndScale computes them
void UpdateViewportConstants() {
    auto& v = g_state.viewport;
    g_state.zscale = ZScale(g_state.zs);
    float hw = v.w * 0.5f, hh = v.h * 0.5f;
    if (!g_own_constants) {
        g_state.vsc[58] = {hw, -hh, (v.max_z - v.min_z) * g_state.zscale, 0};
        g_state.vsc[59] = {v.x + g_state.ss_offset_x + hw, v.y + g_state.ss_offset_y + hh, v.min_z * g_state.zscale, 0};
    }
    // the viewport as the library's SetViewport stores it, for inlined readers
    if (XDK_CDevice_Viewport != xdk::kUnknown) {
        uint32_t p = kDevice + XDK_CDevice_Viewport;
        W32(p, v.x);
        W32(p + 4, v.y);
        W32(p + 8, v.w);
        W32(p + 12, v.h);
        std::memcpy(HostPtr(p + 16), &v.min_z, 4);
        std::memcpy(HostPtr(p + 20), &v.max_z, 4);
    }
}

void SetComplex(uint32_t rs, uint32_t v) {
    g_state.rs[rs] = v;
    W32(kRenderState + rs * 4, v);
}

// The library's default render and texture stage states; its pushes land in the scratch buffer
void ApplyDefaultStates(X86Context& ctx) {
#if HAS_D3D_InitDefaultState
    GUEST_CALL(D3D_InitDefaultState, ctx, {});
#else
    (void)ctx;
    LOG_FIRST(1, "d3d: the image has no D3D_InitDefaultState, default states were not applied");
#endif
}

constexpr bool kVBlank = XDK_CDevice_VBlankCount != xdk::kUnknown && XDK_CDevice_VBlankEvent != xdk::kUnknown &&
                         XDK_CDevice_VBlankCallback != xdk::kUnknown;

// what the library's vertical blank interrupt leaves for the title: count, event, callback
void VerticalBlank(X86Context& ctx) {
    using clock = std::chrono::steady_clock;
    uint32_t info = mem::AllocSystem(12), shown = 0;
    auto next = clock::now();
    for (;;) {
        next = std::max(next + std::chrono::microseconds(16667), clock::now());
        std::this_thread::sleep_until(next);
        uint32_t count = Dev(XDK_CDevice_VBlankCount) + 1, swaps = Dev(kDevSwapCount);
        SetDev(XDK_CDevice_VBlankCount, count);
        ObjectRef event = FromPointer(kDevice + XDK_CDevice_VBlankEvent);
        SetEvent(*event);
        if (uint32_t callback = Dev(XDK_CDevice_VBlankCallback)) {
            // blank count, swaps shown, 1 when this blank showed one
            W32(info, count);
            W32(info + 4, swaps);
            W32(info + 8, swaps != shown);
            CallGuest(ctx, callback, {info});
        }
        shown = swaps;
        // the title clears the event in its own memory before a wait, which the host object does not see
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ResetEvent(*event);
    }
}

}  // namespace

gpu::State& State() { return g_state; }
std::recursive_mutex& Lock() { return g_lock; }

}  // namespace d3d

using namespace d3d;

// device

HLE_STDCALL(Direct3D_CreateDevice, 6) {
    XDK_REQUIRE("d3d", XDK_DATA(D3D_Device), XDK_DATA(D3D_pDevice), XDK_DATA(D3D_DirtyFlags),
                XDK_DATA(D3D_DeferredTextureState), XDK_DATA(D3D_RenderState), XDK_DATA(D3D_Streams),
                XDK_DATA(D3D_SimpleRenderStateEncode));
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Push), XDK_KEY(CDevice_PushThreshold), XDK_KEY(CDevice_BaseVertex),
                XDK_KEY(CDevice_IndexBuffer), XDK_KEY(CDevice_ZScale), XDK_KEY(CDevice_SuperSampleX),
                XDK_KEY(CDevice_SuperSampleY), XDK_KEY(CDevice_Transforms), XDK_KEY(CDevice_Textures),
                XDK_KEY(CDevice_RenderTarget), XDK_KEY(CDevice_Depth), XDK_KEY(CDevice_BackBuffer),
                XDK_KEY(CDevice_FrontBuffer), XDK_KEY(CDevice_SwapCount), XDK_KEY(CDevice_Size),
                XDK_KEY(DirtyFlags_Texture), XDK_KEY(TextureState_Stage), XDK_KEY(Stream_Size),
                XDK_KEY(Stream_Stride), XDK_KEY(Stream_Buffer));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t pp = a[4];  // D3DPRESENT_PARAMETERS
    uint32_t w = R32(pp), h = R32(pp + 4), fmt = R32(pp + 8), count = std::max<uint32_t>(R32(pp + 12), 1);
    uint32_t auto_depth = R32(pp + 0x20), depth_fmt = R32(pp + 0x24), interval = R32(pp + 0x30);
    if (!w) w = 640;
    if (!h) h = 480;
    LOGI("d3d: CreateDevice %ux%u format %X buffers %u depth %u format %X interval %X flags %X", w, h, fmt, count,
         auto_depth, depth_fmt, interval, R32(pp + 0x28));
    std::memset(HostPtr(kDevice), 0, XDK_CDevice_Size);
    W32(kDevicePtr, kDevice);
    ScratchPush();

    uint32_t color_fmt = gpu::LinearColorFormat(fmt);
    g_buffer_count = std::min<uint32_t>(count + 1, 3);
    for (uint32_t i = 0; i < g_buffer_count; i++) g_buffers[i] = NewSurface(w, h, color_fmt, 4);
    g_depth = 0;
    if (auto_depth) {
        uint32_t zf = gpu::LinearDepthFormat(depth_fmt);
        g_depth = NewSurface(w, h, zf, zf == 0x30 || zf == 0x31 ? 2 : 4);  // 0x30 LIN_D16, 0x31 LIN_F16
    }
    if (!g_backend) {
        g_backend = gpu::CreateBackend();
        if (!g_backend->Init(w, h)) LOGE("d3d: renderer init failed");
        if (kRenderThread) g_backend = gpu::CreateThreadedBackend(std::move(g_backend));
    }
    g_state = gpu::State{};
    g_state.viewport = {0, 0, w, h, 0.0f, 1.0f};
    g_interval = interval;  // the swap paces to it while the title leaves D3DRS_PRESENTATIONINTERVAL at 0
    SetDev(kDevRenderTarget, g_buffers[1]);
    SetDev(kDevDepth, g_depth);
    SetDev(kDevBackBuffer, g_buffers[1]);
    SetDev(kDevFrontBuffer, g_buffers[1]);
    float one = 1.0f, zs = g_depth && (R32(g_depth + 12) >> 8 & 0xFF) == 0x30 ? 65535.0f : 16777215.0f;
    std::memcpy(HostPtr(kDevice + kDevSuperX), &one, 4);
    std::memcpy(HostPtr(kDevice + kDevSuperY), &one, 4);
    std::memcpy(HostPtr(kDevice + kDevZScale), &zs, 4);
    // fields the library's init sets that inlined library code and the default states read
    if (XDK_CDevice_AutoDepth != xdk::kUnknown) SetDev(XDK_CDevice_AutoDepth, g_depth);
    if (XDK_CDevice_BackBufferScale != xdk::kUnknown) {
        std::memcpy(HostPtr(kDevice + XDK_CDevice_BackBufferScale), &one, 4);
        std::memcpy(HostPtr(kDevice + XDK_CDevice_BackBufferScale + 4), &one, 4);
    }
    // 60 Hz, widescreen
    if (XDK_CDevice_DisplayFlags != xdk::kUnknown) SetDev(XDK_CDevice_DisplayFlags, 0x400000 | (app::Widescreen() ? 0x10000 : 0));
#if HAS_D3D_FvfShader
    if (XDK_CDevice_VertexShader != xdk::kUnknown) SetDev(XDK_CDevice_VertexShader, SYM(D3D_FvfShader));
#endif
    if (kVBlank) {
        // notification event, signalled
        SetDev(XDK_CDevice_VBlankEvent, 0x00040000);
        SetDev(XDK_CDevice_VBlankEvent + 4, 1);
        static bool started = false;
        if (!started) StartHostThread("vertical blank", VerticalBlank);
        started = true;
    }
    g_state.rt = SurfaceOf(g_buffers[1]);
    g_state.zs = SurfaceOf(g_depth);
    UpdateViewportConstants();
    ApplyDefaultStates(ctx);
    if (a[5]) W32(a[5], kDevice);
    return 0;
}

HLE_STDCALL(D3DDevice_Release, 0) { return 1; }

// synchronisation, the renderer never runs behind

HLE_STDCALL(D3D_BlockOnResource, 1) {
    uint32_t res = a[0];
    if (res && g_backend) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        uint32_t type = R32(res) & 0x70000;  // 0x40000 texture, 0x50000 surface
        if (type == 0x40000 || type == 0x50000) {
            uint32_t data = mem::VirtualFromPhysical(R32(res + 4));
            // perf diagnostic: who blocks on which surface, once each
            static std::unordered_set<uint32_t> seen;
            if (diag::Perf() && seen.insert(data).second) {
                uint32_t code[4] = {}, n = 0;
                for (uint32_t i = 0; i < 32 && n < 4; i++) {
                    uint32_t v = R32(ctx.esp + i * 4);
                    if (v >= X86CodeBase && v < X86CodeEnd) code[n++] = v;
                }
                LOGI("d3d: BlockOnResource on surface %08X, stack %08X %08X %08X %08X", data, code[0], code[1], code[2], code[3]);
            }
            g_backend->Sync(data, 0);
        } else {
            g_backend->WaitResource(res);
        }
    }
    gpu::MarkDirty(res);
    return 0;
}
// lock(resource, face, level, locked, rect, flags): D3DLOCK_NOOVERWRITE skips the block above, yet a
// title may read what the GPU drew
HLE_WRAP(Lock2DSurface, 6) {
    uint32_t res = a[0];
    if ((a[5] & 0x20) && res && g_backend) {  // D3DLOCK_NOOVERWRITE
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_backend->SyncLate(mem::VirtualFromPhysical(R32(res + 4)));
    }
}
HLE_STDCALL(D3DResource_BlockUntilNotBusy, 1) {
    if (a[0] && g_backend) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_backend->WaitResource(a[0]);
    }
    gpu::MarkDirty(a[0]);
    return 0;
}
HLE_STDCALL(D3DResource_IsBusy, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t res = a[0];
    if (!res) return 0;
    // set on the device, a surface also through its texture
    uint32_t common = R32(res), parent = (common & 0x70000) == 0x50000 ? R32(res + 0x14) : 0;
    if ((common & 0x780000) || (parent && (R32(parent) & 0x780000))) return 1;  // internal reference count
    return g_backend && g_backend->Busy(res) ? 1 : 0;
}
HLE_STDCALL(D3D_BlockOnTime, 2) { return 0; }
HLE_STDCALL(D3DDevice_BlockOnFence, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->WaitFence(a[0]);
    return 0;
}
HLE_STDCALL(D3DDevice_InsertFence, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend ? g_backend->Fence() : 0;
}
HLE_STDCALL(D3DDevice_IsFencePending, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend && g_backend->FencePending(a[0]) ? 1 : 0;
}
HLE_STDCALL(D3D_SetFence, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return g_backend ? g_backend->Fence() : 0;
}
HLE_STDCALL(D3D_KickOffAndWaitForIdle, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->Flush();
    return 0;
}
HLE_FASTCALL(CDevice_KickOff, 0) { return 0; }

HLE_STDCALL(D3DDevice_MakeSpace, 0) {
    LOG_FIRST(16, "d3d: MakeSpace from %08X, some caller still pushes hardware commands", R32(ctx.esp));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return ScratchPush();
}
HLE_STDCALL(D3D_MakeRequestedSpace, 2) {
    LOG_FIRST(16, "d3d: MakeRequestedSpace from %08X", R32(ctx.esp));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    return ScratchPush();
}

HLE_STDCALL(D3DDevice_BlockUntilVerticalBlank, 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    return 0;
}
HLE_STDCALL(D3DDevice_SetFlickerFilter, 1) { return 0; }
HLE_STDCALL(D3DDevice_SetSoftDisplayFilter, 1) { return 0; }
HLE_STDCALL(D3DDevice_EnableOverlay, 1) { return 0; }

// The overlay (movie players show YUY2 frames on it) is not composited with the frame buffer: each update is
// converted into a back buffer sized surface of its own and presented as the whole picture.
namespace {
uint32_t g_overlay = 0;  // that surface

uint8_t Clamp255(int v) { return uint8_t(std::clamp(v, 0, 255)); }
}  // namespace

// (surface, source RECT, destination RECT, colour key on, colour key): YUY2 to BGRA on the CPU, then presented
HLE_STDCALL(D3DDevice_UpdateOverlay, 5) {
    uint32_t surf = a[0];
    if (!surf || !g_backend) return 0;
    app::SplashFirstFrame();  // a movie before the first Swap: the game window opens for it, not behind the splash
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t size = R32(surf + 16), sw = (size & 0xFFF) + 1, sh = (size >> 12 & 0xFFF) + 1;
    uint32_t spitch = ((size >> 24 & 0xFF) + 1) * 64;
    const uint8_t* src = static_cast<const uint8_t*>(HostPtr(mem::VirtualFromPhysical(R32(surf + 4))));
    uint32_t back = R32(g_buffers[0] + 16), w = (back & 0xFFF) + 1, h = (back >> 12 & 0xFFF) + 1;
    if (!g_overlay) g_overlay = NewSurface(w, h, (R32(g_buffers[0] + 12) >> 8) & 0xFF, 4);  // the back buffers' format
    uint32_t pitch = ((R32(g_overlay + 16) >> 24 & 0xFF) + 1) * 64;
    uint8_t* dst = static_cast<uint8_t*>(HostPtr(mem::VirtualFromPhysical(R32(g_overlay + 4))));
    auto rect = [](uint32_t r, int32_t def_w, int32_t def_h, int32_t out[4]) {
        if (r) {
            for (int i = 0; i < 4; i++) out[i] = int32_t(R32(r + i * 4));
        } else {
            out[0] = out[1] = 0;
            out[2] = def_w;
            out[3] = def_h;
        }
    };
    // the destination is left out: titles place the overlay inside the TV safe area of their own frame, while
    // the presented surface is the whole picture, the movie fills it
    int32_t sr[4], dr[4];
    rect(a[1], int32_t(sw), int32_t(sh), sr);
    rect(0, int32_t(w), int32_t(h), dr);
    int32_t dw = std::max(dr[2] - dr[0], 1), dh = std::max(dr[3] - dr[1], 1);
    for (uint32_t y = 0; y < h; y++) {
        uint32_t* row = reinterpret_cast<uint32_t*>(dst + size_t(y) * pitch);
        for (uint32_t x = 0; x < w; x++) {
            int32_t ux = int32_t(x) - dr[0], uy = int32_t(y) - dr[1];
            if (ux < 0 || uy < 0 || ux >= dw || uy >= dh) {
                row[x] = 0xFF000000;
                continue;
            }
            uint32_t px = std::min<uint32_t>(uint32_t(sr[0] + ux * (sr[2] - sr[0]) / dw), sw - 1);
            uint32_t py = std::min<uint32_t>(uint32_t(sr[1] + uy * (sr[3] - sr[1]) / dh), sh - 1);
            const uint8_t* q = src + size_t(py) * spitch + (px & ~1u) * 2;  // Y0 U Y1 V
            int yv = q[px & 1 ? 2 : 0] - 16, u = q[1] - 128, v = q[3] - 128;
            int c = 298 * yv + 128;
            row[x] = 0xFF000000u | Clamp255((c + 409 * v) >> 8) << 16 | Clamp255((c - 100 * u - 208 * v) >> 8) << 8 |
                     Clamp255((c + 516 * u) >> 8);
        }
    }
    gpu::MarkDirty(g_overlay);  // written here, not through a lock: the target takes the new pixels
    g_backend->Present(SurfaceOf(g_overlay));
    return 0;
}
HLE_STDCALL(D3DDevice_GetOverlayUpdateStatus, 0) { return 1; }
// the front buffer, read back at the guest size, is kept for the next instance after a quick reboot, which may
// fade from it at its first frames
HLE_STDCALL(D3DDevice_PersistDisplay, 0) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t front = g_buffers[0], size = R32(front + 0x10);
    uint32_t pitch = ((size >> 24 & 0xFF) + 1) * 64, height = (size >> 12 & 0xFFF) + 1, bytes = pitch * height;
    uint32_t data = mem::VirtualFromPhysical(R32(front + 4));
    if (g_backend) g_backend->Sync(data, bytes);
    kernel::SavePersistedDisplay(R32(front), R32(front + 0xC), size, HostPtr(data), bytes);
    LOGI("d3d: PersistDisplay, %u bytes", bytes);
    return 0;
}
HLE_STDCALL(D3DDevice_FlushVertexCache, 0) { return 0; }

HLE_STDCALL(D3DDevice_SetGammaRamp, 2) {
    uint8_t ramp[768];
    std::memcpy(ramp, HostPtr(a[1]), 768);
    if (g_backend) g_backend->SetGamma(ramp);
    return 0;
}

// targets and presentation

HLE_STDCALL(D3DDevice_SetRenderTarget, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));  // a colour offset the title pushed before this belongs to the old target
    uint32_t rt = a[0] ? a[0] : Dev(kDevRenderTarget), zs = a[1];
    AddRef(ctx, rt);
    uint32_t old_rt = Dev(kDevRenderTarget), old_zs = Dev(kDevDepth);
    SetDev(kDevRenderTarget, rt);
    SetDev(kDevDepth, zs);
    AddRef(ctx, zs);
    Release(ctx, old_rt);
    Release(ctx, old_zs);
    g_state.rt = SurfaceOf(rt);
    g_state.zs = SurfaceOf(zs);
    // SetRenderTarget resets the viewport to the whole target
    uint32_t w, h;
    gpu::SurfaceSize(g_state.rt, w, h);
    g_state.viewport = {0, 0, w, h, 0.0f, 1.0f};
    UpdateViewportConstants();
    return 0;
}

HLE_STDCALL(D3DDevice_GetBackBuffer2, 1) {
    int32_t i = int32_t(a[0]);
    uint32_t s = i < 0 ? g_buffers[0] : g_buffers[std::min<uint32_t>(uint32_t(i) + 1, g_buffer_count - 1)];
    W32(s, R32(s) + 1);
    return s;
}

HLE_STDCALL(D3DDevice_GetRenderTarget2, 0) {
    uint32_t s = Dev(kDevRenderTarget);
    if (s) W32(s, R32(s) + 1);
    return s;
}

HLE_STDCALL(D3DDevice_GetDepthStencilSurface2, 0) {
    uint32_t s = Dev(kDevDepth);
    if (s) W32(s, R32(s) + 1);
    return s;
}

namespace {
std::atomic<uint32_t> g_presents{0};
}  // namespace

HLE_STDCALL(D3DDevice_Swap, 1) {
    app::WaitWhileUnfocused();
    app::SplashFirstFrame();
    // handed over from the previous instance: the window opens on the second frame, the first picture with
    // anything drawn over it right after
    if (app::HandingOver() && g_presents >= 2) {
        {
            std::lock_guard<std::recursive_mutex> lock(g_lock);
            if (g_backend) g_backend->Flush();
        }
        app::FinishHandover();
    }
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_presents++;
    // the kept display is shown until the title presents its own
    kernel::SetSavedDataAddress(0);
    // the ramp an inlined SetGammaRamp left for the vertical blank that shows this swap
    if (g_backend && XDK_CDevice_GammaPending != xdk::kUnknown && XDK_CDevice_GammaRamps != xdk::kUnknown) {
        uint32_t i = Dev(kDevSwapCount) & 1;
        if (Dev(XDK_CDevice_GammaPending + i * 4) == 1) {
            SetDev(XDK_CDevice_GammaPending + i * 4, 0);
            g_backend->SetGamma(static_cast<const uint8_t*>(HostPtr(kDevice + XDK_CDevice_GammaRamps + i * 0x300)));
        }
    }
    Walk(Dev(kDevPush));  // reports asked for at the end of the frame
    if (g_backend) g_backend->Present(SurfaceOf(g_buffers[1]));
    // D3DRS_PRESENTATIONINTERVAL from the library's table: a title whose logic counts frames (30 fps under
    // D3DPRESENT_INTERVAL_TWO) keeps its speed; IMMEDIATE (bit 31) and DEFAULT (0) wait one vblank at most
    uint32_t interval = R32(kRenderState + gpu::kRsPresentationInterval * 4);
    if (!interval) interval = g_interval;
    LOG_FIRST(8, "d3d: presentation interval %X", interval);
    app::Pace(interval & 0x80000000u ? 0 : std::clamp<uint32_t>(interval & 0xF, 1, 4));
    // like the XDK, the surfaces stay put and their memory rotates: titles keep back buffer pointers and copy its data
    uint32_t last = g_buffers[g_buffer_count - 1], data = R32(last + 4), lk = R32(last + 8);
    for (uint32_t i = g_buffer_count - 1; i > 0; i--) {
        W32(g_buffers[i] + 4, R32(g_buffers[i - 1] + 4));
        W32(g_buffers[i] + 8, R32(g_buffers[i - 1] + 8));
    }
    W32(g_buffers[0] + 4, data);
    W32(g_buffers[0] + 8, lk);
    g_state.rt = SurfaceOf(Dev(kDevRenderTarget));
    uint32_t n = Dev(kDevSwapCount) + 1;
    SetDev(kDevSwapCount, n);
    return n;
}

HLE_STDCALL(D3DDevice_Clear, 6) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    const uint8_t* rects = a[0] && a[1] ? static_cast<const uint8_t*>(HostPtr(a[1])) : nullptr;
    if (g_backend) g_backend->Clear(g_state, a[0], rects, a[2], a[3], ArgF(a, 4), a[5]);
    return 0;
}

HLE_STDCALL(D3DDevice_CopyRects, 5) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_backend) g_backend->Copy(SurfaceOf(a[0]), SurfaceOf(a[3]), a[2], a[1], a[4]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetViewport, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t v = a[0];
    g_state.viewport = {R32(v), R32(v + 4), R32(v + 8), R32(v + 12), GF(v + 16), GF(v + 20)};
    g_state.scissor_count = 0;
    UpdateViewportConstants();
    return 0;
}

HLE_STDCALL(D3DDevice_SetScissors, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_state.scissor_count = std::min<uint32_t>(a[0], 8);
    for (uint32_t i = 0; i < g_state.scissor_count * 4; i++) g_state.scissors[i] = R32(a[2] + i * 4);
    return 0;
}

HLE_STDCALL(D3DDevice_SetScreenSpaceOffset, 2) {
    g_state.ss_offset_x = ArgF(a, 0);
    g_state.ss_offset_y = ArgF(a, 1);
    UpdateViewportConstants();
    return 0;
}

// render states

HLE_FASTCALL(D3DDevice_SetRenderState_Simple, 0) {
    uint32_t method = a[0] & 0x1FFC, value = a[1];
    g_state.simple[method >> 2] = value;
    return 0;
}

HLE_STDCALL(D3DDevice_SetRenderStateNotInline, 2) {
    uint32_t rs = a[0], v = a[1];
    if (rs < gpu::kRsDeferredFirst) {
        uint32_t method = R32(kSimpleEncode + rs * 4) & 0x1FFC;
        g_state.simple[method >> 2] = v;
        W32(kRenderState + rs * 4, v);  // titles read their states back from the library's table
    } else if (rs < gpu::kRsMax) {
        SetComplex(rs, v);
    }
    return 0;
}

#define COMPLEX_STATE(name, rs)                       \
    HLE_STDCALL(D3DDevice_SetRenderState_##name, 1) { \
        SetComplex(rs, a[0]);                         \
        return 0;                                     \
    }
HLE_STDCALL(D3DDevice_SetRenderState_PSTextureModes, 1) {
    SetComplex(gpu::kRsPsTextureModes, a[0]);
    return 0;
}
COMPLEX_STATE(VertexBlend, gpu::kRsVertexBlend)
COMPLEX_STATE(FogColor, gpu::kRsFogColor)
COMPLEX_STATE(FillMode, gpu::kRsFillMode)
COMPLEX_STATE(BackFillMode, gpu::kRsBackFillMode)
COMPLEX_STATE(TwoSidedLighting, gpu::kRsTwoSidedLighting)
COMPLEX_STATE(NormalizeNormals, gpu::kRsNormalizeNormals)
COMPLEX_STATE(ZEnable, gpu::kRsZEnable)
COMPLEX_STATE(StencilEnable, gpu::kRsStencilEnable)
COMPLEX_STATE(StencilFail, gpu::kRsStencilFail)
COMPLEX_STATE(FrontFace, gpu::kRsFrontFace)
COMPLEX_STATE(CullMode, gpu::kRsCullMode)
COMPLEX_STATE(TextureFactor, gpu::kRsTextureFactor)
COMPLEX_STATE(LogicOp, gpu::kRsLogicOp)
COMPLEX_STATE(EdgeAntiAlias, gpu::kRsEdgeAntiAlias)
COMPLEX_STATE(MultiSampleAntiAlias, gpu::kRsMultiSampleAntiAlias)
COMPLEX_STATE(MultiSampleMask, gpu::kRsMultiSampleMask)
COMPLEX_STATE(MultiSampleMode, gpu::kRsMultiSampleMode)
COMPLEX_STATE(MultiSampleRenderTargetMode, gpu::kRsMultiSampleRenderTargetMode)
COMPLEX_STATE(ShadowFunc, gpu::kRsShadowFunc)
COMPLEX_STATE(LineWidth, gpu::kRsLineWidth)
COMPLEX_STATE(SampleAlpha, gpu::kRsSampleAlpha)
COMPLEX_STATE(Dxt1NoiseEnable, gpu::kRsDxt1NoiseEnable)
COMPLEX_STATE(YuvEnable, gpu::kRsYuvEnable)
COMPLEX_STATE(OcclusionCullEnable, gpu::kRsOcclusionCullEnable)
COMPLEX_STATE(StencilCullEnable, gpu::kRsStencilCullEnable)
COMPLEX_STATE(RopZCmpAlwaysRead, gpu::kRsRopZCmpAlwaysRead)
COMPLEX_STATE(RopZRead, gpu::kRsRopZRead)
COMPLEX_STATE(DoNotCullUncompressed, gpu::kRsDoNotCullUncompressed)

HLE_STDCALL(D3DDevice_SetTextureState_TexCoordIndex, 2) {
    g_state.tss[a[0] & 3][gpu::kTssTexCoordIndex] = a[1];
    W32(kTextureState + ((a[0] & 3) * kStageStates + gpu::kTssTexCoordIndex) * 4, a[1]);
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_BumpEnv, 3) {
    if (a[1] >= gpu::kTssBumpEnvMat00 && a[1] <= gpu::kTssBumpEnvLOffset) g_state.tss[a[0] & 3][a[1]] = a[2];
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_BorderColor, 2) {
    g_state.tss[a[0] & 3][gpu::kTssBorderColor] = a[1];
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureState_ColorKeyColor, 2) {
    g_state.tss[a[0] & 3][gpu::kTssColorKeyColor] = a[1];
    return 0;
}
HLE_STDCALL(D3DDevice_SetTextureStageStateNotInline, 3) {
    uint32_t stage = a[0] & 3, type = a[1], v = a[2];
    if (type < gpu::kTssCount) {
        g_state.tss[stage][type] = v;
        W32(kTextureState + (stage * kStageStates + type) * 4, v);
    }
    return 0;
}

// the XDK's transform types: view 0, projection 1, texture 2-5, world 6-9; the renderer keeps Direct3D 8's numbers
// (view 2, projection 3, texture 16, world 256). The device copy is what GetTransform reads.
HLE_STDCALL(D3DDevice_SetTransform, 2) {
    uint32_t t = a[0];
    if (t >= 10) return 0;
    std::memcpy(HostPtr(kDevice + kDevTransforms + t * 64), HostPtr(a[1]), 64);
    uint32_t pc = t < 2 ? t + 2 : t < 6 ? t + 14 : t + 250;
    std::memcpy(g_state.transforms[pc].data(), HostPtr(a[1]), 64);
    return 0;
}

HLE_STDCALL(D3DDevice_SetMaterial, 1) {
    std::memcpy(g_state.material.data(), HostPtr(a[0]), 17 * 4);
    return 0;
}

// the XDK keeps its own light lists, the renderer gets a copy
HLE_WRAP(D3DDevice_SetLight, 2) {
    uint32_t i = a[0], l = a[1];
    if (i < 8 && l) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        std::memcpy(g_state.lights[i].data(), HostPtr(l), 26 * 4);
    }
}
HLE_WRAP(D3DDevice_LightEnable, 2) {
    uint32_t i = a[0], on = a[1];
    if (i < 8) {
        std::lock_guard<std::recursive_mutex> lock(g_lock);
        g_state.light_enable = on ? g_state.light_enable | 1u << i : g_state.light_enable & ~(1u << i);
    }
}

// resources in use

HLE_STDCALL(D3DDevice_SetTexture, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t stage = a[0] & 3, tex = a[1];
    uint32_t old = Dev(kDevTextures + stage * 4);
    AddRef(ctx, tex);
    SetDev(kDevTextures + stage * 4, tex);
    Release(ctx, old);
    g_state.textures[stage] = tex;
    W32(kDirtyFlags, R32(kDirtyFlags) | XDK_DirtyFlags_Texture);
    return 0;
}

// SetPalette(stage, palette): the colours a P8 texture of the stage looks up
HLE_STDCALL(D3DDevice_SetPalette, 2) {
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Palettes));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t stage = a[0] & 3, pal = a[1];
    uint32_t old = Dev(kDevPalettes + stage * 4);
    AddRef(ctx, pal);
    SetDev(kDevPalettes + stage * 4, pal);
    Release(ctx, old);
    g_state.palettes[stage] = pal;
    return 0;
}

// SwitchTexture(method, data, format), fastcall: a swizzled texture of the bound one's shape replaces it. It is
// bound through a header of the stage's own holding the bound texture's fields with the new data and format.
HLE_FASTCALL(D3DDevice_SwitchTexture, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    static uint32_t headers[4];
    // texture methods from 0x1B00, 0x40 per stage
    uint32_t stage = (((a[0] & 0x1FFC) - 0x1B00) >> 6) & 3, cur = g_state.textures[stage];
    uint32_t& h = headers[stage];
    if (!h) h = mem::AllocSystem(20);
    if (cur && cur != h) std::memcpy(HostPtr(h), HostPtr(cur), 20);
    W32(h + 4, a[1]);
    W32(h + 12, a[2]);
    g_state.textures[stage] = h;
    return 0;
}

// the API's streams and shader replace what the title pushed itself (the library sends them at the next draw):
// the commands pushed so far are read first, a later push of arrays takes over again
static void UseApiVertexState() {
    Walk(Dev(kDevPush));
    g_raw.arrays = false;
}

HLE_STDCALL(D3DDevice_SetStreamSource, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    UseApiVertexState();
    uint32_t n = a[0] & 15, vb = a[1], stride = a[2];
    uint32_t old = R32(kStreams + n * kStreamSize + XDK_Stream_Buffer);
    AddRef(ctx, vb);
    W32(kStreams + n * kStreamSize + XDK_Stream_Stride, stride);
    W32(kStreams + n * kStreamSize + XDK_Stream_Buffer, vb);
    Release(ctx, old);
    g_state.streams[n] = {vb, stride};
    return 0;
}

HLE_STDCALL(D3DDevice_SetIndices, 2) {
    XDK_REQUIRE("d3d", XDK_DATA(D3D_IndexData));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t ib = a[0];
    uint32_t old = Dev(kDevIndexBuffer);
    AddRef(ctx, ib);
    W32(kIndexData, ib ? R32(ib + 4) : 0);
    SetDev(kDevIndexBuffer, ib);
    SetDev(kDevBaseVertex, a[1]);
    Release(ctx, old);
    g_state.index_buffer = ib;
    g_state.index_data = ib ? R32(ib + 4) : 0;
    g_state.base_vertex = a[1];
    return 0;
}

// shaders

HLE_STDCALL(D3DDevice_CreateVertexShader, 4) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t decl = a[0], func = a[1], out = a[2];
    auto vs = std::make_shared<gpu::VertexShader>();
    if (decl) {
        for (uint32_t i = 0; i < 1024; i++) {
            uint32_t t = R32(decl + i * 4);
            vs->decl.push_back(t);
            if (t == 0xFFFFFFFF) break;
        }
    }
    if (func) {
        uint32_t hdr = R32(func);
        vs->program_type = hdr & 0xFFFF;
        uint32_t n = hdr >> 16;
        for (uint32_t i = 0; i < n * 4; i++) vs->program.push_back(R32(func + 4 + i * 4));
    }
    gpu::ParseDeclaration(*vs);
    uint32_t rec = mem::AllocSystem(16, 16);
    vs->handle = rec | 1;
    g_vs[vs->handle] = vs;
    W32(out, vs->handle);
    LOG_FIRST(64, "d3d: vertex shader %08X, %zu decl tokens, %zu instructions, type %X", vs->handle, vs->decl.size(),
              vs->program.size() / 4, vs->program_type);
    return 0;
}

HLE_STDCALL(D3DDevice_DeleteVertexShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto it = g_vs.find(a[0]);
    if (it != g_vs.end()) {
        mem::FreeSystem(a[0] & ~1u);
        g_vs.erase(it);
    }
    return 0;
}

// (handle, data, size in and out): the program as it was given, the size alone without data or room
HLE_STDCALL(D3DDevice_GetVertexShaderFunction, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto it = g_vs.find(a[0]);
    uint32_t words = it == g_vs.end() ? 0 : uint32_t(it->second->program.size());
    uint32_t need = words ? 4 + words * 4 : 0, room = R32(a[2]);
    W32(a[2], need);
    if (need > room) return a[1] ? 0x88760867 : 0;  // D3DERR_MOREDATA
    if (!a[1] || !need) return 0;
    W32(a[1], it->second->program_type | words / 4 << 16);
    for (uint32_t i = 0; i < words; i++) W32(a[1] + 4 + i * 4, it->second->program[i]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetVertexShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    UseApiVertexState();
    g_raw.selected = a[0];
    g_raw.own_program = false;
    uint32_t h = a[0];
    g_state.vs_handle = h;
    g_state.vs = nullptr;
    if (h & 1) {
        auto it = g_vs.find(h);
        if (it != g_vs.end()) {
            g_state.vs = it->second;
            // constants declared with the shader load when it is selected
            for (auto& [reg, v] : it->second->constants)
                if (reg < 192) g_state.vsc[reg] = v;
        } else {
            LOG_FIRST(8, "d3d: SetVertexShader with unknown handle %08X", h);
        }
    }
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstant1Fast, 0) {
    uint32_t reg = a[0];
    if (reg < 192) std::memcpy(g_state.vsc[reg].data(), HostPtr(a[1]), 16);
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstant1, 0) {
    uint32_t reg = a[0];
    if (reg < 192) std::memcpy(g_state.vsc[reg].data(), HostPtr(a[1]), 16);
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstant4, 0) {
    uint32_t reg = a[0], src = a[1];
    for (uint32_t i = 0; i < 4 && reg + i < 192; i++) std::memcpy(g_state.vsc[reg + i].data(), HostPtr(src + i * 16), 16);
    return 0;
}

HLE_FASTCALL(D3DDevice_SetVertexShaderConstantNotInlineFast, 1) {
    uint32_t reg = a[0], src = a[1], dwords = a[2];
    for (uint32_t i = 0; i < dwords / 4 && reg + i < 192; i++) std::memcpy(g_state.vsc[reg + i].data(), HostPtr(src + i * 16), 16);
    return 0;
}

HLE_STDCALL(D3DDevice_LoadVertexShader, 2) {
    LOG_FIRST(4, "d3d: LoadVertexShader %08X at %u is not supported", a[0], a[1]);
    return 0;
}
// (handle, slot): the title's own program memory runs from the slot; its arrays are pointed by its own commands
HLE_STDCALL(D3DDevice_SelectVertexShader, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    g_raw.start = a[1];
    g_raw.selected = a[0];
    g_raw.own_program = true;
    static const bool log = std::getenv("STEAL_GEOM_LOG") != nullptr;
    static std::unordered_set<uint64_t> seen;
    if (log && seen.insert(uint64_t(a[0]) << 32 | a[1]).second)
        LOGE("select vertex shader %08X at %u from %05X, arrays %d", a[0], a[1], R32(ctx.esp), int(g_raw.arrays));
    if (log && DirectHandle(a[0])) {
        static std::unordered_set<std::string> tables;
        std::string t;
        char buf[48];
        for (uint32_t r = 0; r < 16; r++) {
            uint32_t at = (a[0] & ~1u) + 0x14 + r * 16;
            if ((R32(at + 8) & 0xFF) == 0x02) continue;
            snprintf(buf, sizeof(buf), " v%u s%u +%u t%X", r, R32(at), R32(at + 4), R32(at + 8));
            t += buf;
        }
        if (tables.insert(t).second) LOGE("select direct attributes:%s", t.c_str());
    }
    return 0;
}
HLE_STDCALL(D3DDevice_SetShaderConstantMode, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_own_constants = a[0] & 0x10;  // D3DSCM_NORESERVEDCONSTANTS
    UpdateViewportConstants();
    return 0;
}

HLE_STDCALL(D3DDevice_CreatePixelShader, 2) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto ps = std::make_shared<gpu::PixelShader>();
    std::memcpy(ps->def, HostPtr(a[0]), sizeof(ps->def));
    uint32_t rec = mem::AllocSystem(16, 16);
    ps->handle = rec;
    g_ps[rec] = ps;
    W32(a[1], rec);
    return 0;
}

HLE_STDCALL(D3DDevice_DeletePixelShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (g_ps.erase(a[0])) mem::FreeSystem(a[0]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetPixelShader, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_state.ps = nullptr;
    if (!a[0]) return 0;
    auto it = g_ps.find(a[0]);
    if (it != g_ps.end()) {
        g_state.ps = it->second;
    } else if (XDK_PixelShader_Definition != xdk::kUnknown) {
        // made by library code the title inlined: the object points at its definition, handles are reused
        static std::unordered_map<uint32_t, std::shared_ptr<gpu::PixelShader>> made;
        const void* def = HostPtr(R32(a[0] + XDK_PixelShader_Definition));
        auto& ps = made[a[0]];
        if (!ps || std::memcmp(ps->def, def, sizeof(ps->def))) {
            ps = std::make_shared<gpu::PixelShader>();
            ps->handle = a[0];
            std::memcpy(ps->def, def, sizeof(ps->def));
        }
        g_state.ps = ps;
    } else {
        LOG_FIRST(8, "d3d: SetPixelShader with unknown handle %08X", a[0]);
        return 0;
    }
    // the definition loads the combiner state like the XDK pushes it
    const uint32_t* d = g_state.ps->def;
    auto put = [&](uint32_t method, uint32_t v) { g_state.simple[method >> 2] = v; };
    for (int i = 0; i < 8; i++) {
        put(gpu::kMPsAlphaInputs + i * 4, d[0 + i]);
        put(gpu::kMPsConstant0 + i * 4, d[10 + i]);
        put(gpu::kMPsConstant1 + i * 4, d[18 + i]);
        put(gpu::kMPsAlphaOutputs + i * 4, d[26 + i]);
        put(gpu::kMPsRgbInputs + i * 4, d[34 + i]);
        put(gpu::kMPsRgbOutputs + i * 4, d[45 + i]);
    }
    put(gpu::kMPsFinalAbcd, d[8]);
    put(gpu::kMPsFinalEfgh, d[9]);
    put(gpu::kMPsCompareMode, d[42]);
    put(gpu::kMPsFinalConstant0, d[43]);
    put(gpu::kMPsFinalConstant1, d[44]);
    put(gpu::kMPsCombinerCount, d[53]);
    put(gpu::kMPsDotMapping, d[55]);
    put(gpu::kMPsInputTexture, d[56]);
    SetComplex(gpu::kRsPsTextureModes, d[54]);
    return 0;
}

HLE_STDCALL(D3DDevice_SetPixelShaderConstant, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t reg = a[0], src = a[1], n = a[2];
    for (uint32_t i = 0; i < n && reg + i < 16; i++) std::memcpy(g_state.psc[reg + i].data(), HostPtr(src + i * 16), 16);
    static const bool psc_log = std::getenv("XBE_PSC_LOG") != nullptr;
    if (psc_log) {
        const auto& c = g_state.psc[reg & 15];
        LOGE("psc reg %u n %u (%.3f %.3f %.3f %.3f) ps %08X map %08X %08X %08X", reg, n, c[0], c[1], c[2], c[3],
             g_state.ps ? g_state.ps->handle : 0, g_state.ps ? g_state.ps->def[57] : 0,
             g_state.ps ? g_state.ps->def[58] : 0, g_state.ps ? g_state.ps->def[59] : 0);
    }
    gpu::ApplyPixelConstants(g_state, reg, n);
    return 0;
}

// drawing

// STEAL_GEOM_LOG: a small quad list (the lens flare): where its vertices come from, the draw's range, the arrays'
// formats and offsets and the first vertices' words, a few a second
// STEAL_GEOM_LOG: a small quad list drawn over the API's streams: the selected handle and program slot, the vertex
// shader, the streams and the selected declaration, the first vertices' words of stream 0, a few a second
static void LogApiQuads(uint32_t prim, uint32_t start, uint32_t count) {
    static const bool geom_log = std::getenv("STEAL_GEOM_LOG") != nullptr;
    constexpr uint32_t kQuads = 8, kSmall = 64, kShown = 6, kWords = 10, kPerSecond = 6;
    if (!geom_log || g_raw.arrays || prim != kQuads || count > kSmall) return;
    static auto second = std::chrono::steady_clock::now();
    static uint32_t logged = 0;
    auto now = std::chrono::steady_clock::now();
    if (now - second >= std::chrono::seconds(1)) second = now, logged = 0;
    if (++logged > kPerSecond) return;
    std::string line;
    char buf[80];
    for (uint32_t n = 0; n < 4; n++) {
        uint32_t vb = g_state.streams[n].vb;
        snprintf(buf, sizeof(buf), " s%u %08X/%u data %08X", n, vb, g_state.streams[n].stride, vb ? R32(vb + 4) : 0);
        line += buf;
    }
    auto it = g_vs.find(g_raw.selected);
    if (it != g_vs.end())
        for (uint32_t r = 0; r < 16; r++) {
            const auto& in = it->second->inputs[r];
            if (in.stream < 0) continue;
            snprintf(buf, sizeof(buf), " v%u s%d +%u t%X", r, in.stream, in.offset, in.type);
            line += buf;
        }
    LOGE("api quads: start %u count %u vs %08X selected %08X own program %d at %u%s", start, count, g_state.vs_handle,
         g_raw.selected, int(g_raw.own_program), g_raw.start, line.c_str());
    uint32_t vb = g_state.streams[0].vb, stride = g_state.streams[0].stride;
    if (!vb || !stride) return;
    uint32_t data = mem::VirtualFromPhysical(R32(vb + 4));
    for (uint32_t v = 0; v < kShown && v < count; v++) {
        line.clear();
        for (uint32_t w = 0; w < stride / 4 && w < kWords; w++) {
            uint32_t x = R32(data + (start + v) * stride + w * 4);
            float f;
            std::memcpy(&f, &x, 4);
            snprintf(buf, sizeof(buf), " %g", f);
            line += buf;
        }
        LOGE("  vertex %u:%s", start + v, line.c_str());
    }
}

static void LogQuadSource(const char* call, uint32_t prim, uint32_t start, uint32_t count, uint32_t data,
                          uint32_t stride) {
    static const bool geom_log = std::getenv("STEAL_GEOM_LOG") != nullptr;
    constexpr uint32_t kQuads = 8, kSmall = 64, kShown = 12, kWords = 12, kPerSecond = 6;
    constexpr uint32_t kFloat3 = 0x32;  // D3DVSDT_FLOAT3: the flare's positions; the HUD's are FLOAT2
    if (!geom_log || prim != kQuads || count > kSmall) return;
    uint32_t type = g_raw.arrays ? g_raw.format[0] & 0xFF : g_state.vs ? g_state.vs->inputs[0].type & 0xFF : 0;
    if (type != kFloat3) return;
    static auto second = std::chrono::steady_clock::now();
    static uint32_t logged = 0;
    auto now = std::chrono::steady_clock::now();
    if (now - second >= std::chrono::seconds(1)) second = now, logged = 0;
    if (++logged > kPerSecond) return;
    std::string line;
    char buf[64];
    for (uint32_t r = 0; r < 16 && g_raw.arrays; r++) {
        if (!((g_raw.format[r] >> 4) & 0xF)) continue;
        snprintf(buf, sizeof(buf), " v%u fmt %X off %08X", r, g_raw.format[r], g_raw.offset[r]);
        line += buf;
    }
    LOGE("%s quads: start %u count %u raw %d data %08X stride %u%s", call, start, count, int(g_raw.arrays), data,
         stride, line.c_str());
    if (!data || !stride) return;
    for (uint32_t v = 0; v < kShown && v < count; v++) {
        line.clear();
        for (uint32_t w = 0; w < stride / 4 && w < kWords; w++) {
            uint32_t x = R32(data + (start + v) * stride + w * 4);
            float f;
            std::memcpy(&f, &x, 4);
            snprintf(buf, sizeof(buf), " %g", f);
            line += buf;
        }
        LOGE("  vertex %u:%s", start + v, line.c_str());
    }
}

HLE_STDCALL(D3DDevice_DrawVertices, 3) {
    PushStat(0xD0000001u);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));  // a visibility test begun since the last draw counts this one
    SyncDeferred();
    if (g_raw.arrays)
        LogQuadSource("arrays", a[0], a[1], a[2], mem::VirtualFromPhysical(g_raw.offset[0] & 0x7FFFFFFF),
                      g_raw.format[0] >> 8);
    LogApiQuads(a[0], a[1], a[2]);
    if (g_backend) g_backend->Draw(DrawState(), a[0], a[1], a[2], nullptr);
    return 0;
}

HLE_STDCALL(D3DDevice_DrawIndexedVertices, 3) {
    PushStat(0xD0000002u);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    SyncDeferred();
    const uint8_t* indices = a[2] ? static_cast<const uint8_t*>(HostPtr(a[2])) : nullptr;
    if (g_backend) g_backend->Draw(DrawState(), a[0], 0, a[1], indices);
    return 0;
}

// DrawVerticesUP(prim, count, vertices, stride): the vertices are copied to a ring the render thread reads, stream 0
// points there through a vertex buffer header for the draw. A half of the ring is written again once the draws
// that read it are done.
namespace {
constexpr uint32_t kUpRing = 4u << 20, kUpHalf = kUpRing / 2;
uint32_t g_up_ring = 0, g_up_pos = 0, g_up_header = 0, g_up_fence[2] = {};

uint32_t UpVertices(const void* src, uint32_t bytes) {
    if (!g_up_ring) {
        g_up_ring = mem::AllocContiguous(kUpRing, 0, 0xFFFFFFFF, 0x1000);
        g_up_header = mem::AllocSystem(20);
        std::memset(HostPtr(g_up_header), 0, 20);
        W32(g_up_header, 1);  // a vertex buffer, one reference
    }
    bytes = std::min(bytes, kUpHalf);
    uint32_t room = (bytes + 63) & ~63u;
    uint32_t pos = g_up_pos + room > kUpRing ? 0 : g_up_pos;
    uint32_t from = g_up_pos ? (g_up_pos - 1) / kUpHalf : 1, to = (pos + room - 1) / kUpHalf;
    if (to != from) {
        g_up_fence[from] = g_backend->Fence();
        g_backend->WaitFence(g_up_fence[to]);
    }
    std::memcpy(HostPtr(g_up_ring + pos), src, bytes);
    g_up_pos = pos + room;
    W32(g_up_header + 4, mem::PhysicalFromVirtual(g_up_ring + pos));
    return g_up_header;
}

void DrawUp(uint32_t prim, uint32_t count, const void* vertices, uint32_t stride) {
    if (!g_backend || !count || !stride) return;
    SyncDeferred();
    auto stream = g_state.streams[0];
    g_state.streams[0] = {UpVertices(vertices, count * stride), stride};
    g_backend->Draw(g_state, prim, 0, count, nullptr);
    g_state.streams[0] = stream;
}

// bytes per vertex of the bound format: stream 0 of the FVF or of the shader's declaration
uint32_t VertexBytes() {
    if (g_state.vs_handle & 1) return g_state.vs ? g_state.vs->stream_stride[0] : 0;
    gpu::VertexShader fvf;
    gpu::FvfInputs(g_state.vs_handle, fvf);
    return fvf.stream_stride[0];
}

uint32_t g_push_begin = 0;
}  // namespace

// Begin/End immediate mode. Titles inline Begin and the vertex data calls, so their commands are read from the scratch buffer.
namespace d3d {
namespace {
uint32_t g_im_prim = 0, g_im_mask = 0;  // open primitive, slots it wrote
float g_im_attr[64];                    // 16 slots of x y z w
std::vector<float> g_im_vertices;       // every slot per vertex
std::vector<uint32_t> g_im_methods;     // STEAL_GEOM_LOG: the open primitive's methods and values

// visibility tests: depth passing pixels are counted while enabled, a report gets the count since the clear,
// at a physical address in the first 16 MB
void ApplyVisibilityMethod(uint32_t m, uint32_t v) {
    constexpr uint32_t kReportAddressMask = 0xFFFFFF;
    if (m == gpu::kMCountEnable) g_state.simple[gpu::kMCountEnable >> 2] = v;
    if (!g_backend) return;
    if (m == gpu::kMClearReport) g_backend->ClearReport();
    if (m == gpu::kMGetReport) g_backend->GetReport(mem::VirtualFromPhysical(v & kReportAddressMask));
}

// draws the open primitive: the slots it wrote are FLOAT4 inputs of the bound program, or of the fixed pipeline
void EndPrimitive() {
    struct Made {
        std::shared_ptr<gpu::VertexShader> from, vs;
    };
    // the render thread tells shaders apart by pointer: one object per program and slot set, the program kept alive
    static std::map<std::pair<gpu::VertexShader*, uint32_t>, Made> made;
    static std::vector<float> packed;
    uint32_t prim = g_im_prim, count = uint32_t(g_im_vertices.size() / 64), handle = g_state.vs_handle;
    g_im_prim = 0;
    if (!prim || !count) return;
    // STEAL_GEOM_LOG: a quad list's methods as the title pushed them, a few a second
    static const bool geom_log = std::getenv("STEAL_GEOM_LOG") != nullptr;
    constexpr uint32_t kQuads = 8, kPerSecond = 4, kShownMethods = 96;
    if (geom_log && prim == kQuads) {
        static auto second = std::chrono::steady_clock::now();
        static uint32_t logged = 0;
        auto now = std::chrono::steady_clock::now();
        if (now - second >= std::chrono::seconds(1)) second = now, logged = 0;
        if (++logged <= kPerSecond) {
            std::string line;
            char buf[40];
            for (size_t i = 0; i + 1 < g_im_methods.size() && i < kShownMethods * 2; i += 2) {
                float f;
                std::memcpy(&f, &g_im_methods[i + 1], 4);
                snprintf(buf, sizeof(buf), " %X=%g", g_im_methods[i], f);
                line += buf;
            }
            LOGE("immediate quads: %u vertices mask %X:%s", count, g_im_mask, line.c_str());
        }
    }
    if (!(handle & 1) && (handle & 0xE) == 4) LOG_FIRST(1, "d3d: pretransformed immediate mode vertices are not supported");
    auto bound = g_state.vs, from = handle & 1 ? bound : nullptr;
    Made& m = made[{from.get(), g_im_mask}];
    if (!m.vs) {
        m.from = from;
        m.vs = std::make_shared<gpu::VertexShader>();
        if (from) {
            m.vs->program = from->program;
            m.vs->program_type = from->program_type;
        }
        for (uint32_t s = 0; s < 16; s++)
            if (g_im_mask >> s & 1) {
                m.vs->inputs[s] = {0, m.vs->stream_stride[0], 0x42};
                m.vs->stream_stride[0] += 16;
            }
    }
    packed.clear();
    for (uint32_t v = 0; v < count; v++)
        for (uint32_t s = 0; s < 16; s++)
            if (g_im_mask >> s & 1) packed.insert(packed.end(), &g_im_vertices[v * 64 + s * 4], &g_im_vertices[v * 64 + s * 4] + 4);
    g_state.vs = m.vs;
    g_state.vs_handle = handle | 1;
    DrawUp(prim, count, packed.data(), m.vs->stream_stride[0]);
    g_state.vs = bound;
    g_state.vs_handle = handle;
}

// reads the commands pushed since the last call: vertex data between a begin and an end is drawn, visibility tests
// are run, the rest is skipped

// methods of the transform program, constants and vertex arrays; false for any other
bool ApplyRawMethod(uint32_t m, uint32_t v) {
    constexpr uint32_t kProgram = 0x0B00, kConstant = 0x0B80, kArrayOffset = 0x1720, kArrayFormat = 0x1760,
                       kProgramLoad = 0x1E9C, kProgramStart = 0x1EA0, kConstantLoad = 0x1EA4;
    if (m >= kProgram && m < kProgram + 0x80) {
        if (g_raw.load < 136) g_raw.program[g_raw.load * 4 + g_raw.load_word] = v;
        if (++g_raw.load_word == 4) {
            g_raw.load_word = 0;
            g_raw.load++;
        }
    } else if (m >= kConstant && m < kConstant + 0x80) {
        if (g_raw.const_load < 192) {
            float f;
            std::memcpy(&f, &v, 4);
            g_state.vsc[g_raw.const_load][g_raw.const_word] = f;
        }
        if (++g_raw.const_word == 4) {
            g_raw.const_word = 0;
            g_raw.const_load++;
        }
    } else if (m >= kArrayOffset && m < kArrayOffset + 0x40) {
        g_raw.offset[(m - kArrayOffset) / 4] = v;
        g_raw.arrays = true;
    } else if (m >= kArrayFormat && m < kArrayFormat + 0x40) {
        g_raw.format[(m - kArrayFormat) / 4] = v;
        g_raw.arrays = true;
    } else if (m == kProgramLoad) {
        g_raw.load = v;
        g_raw.load_word = 0;
    } else if (m == kProgramStart) {
        g_raw.start = v;
        g_raw.own_program = true;
    } else if (m == kConstantLoad) {
        g_raw.const_load = v;
        g_raw.const_word = 0;
    } else if (m == 0x030C) {  // SET_DEPTH_TEST_ENABLE, what SetRenderState_ZEnable pushes
        g_state.rs[gpu::kRsZEnable] = v;
    } else if (m == 0x032C) {  // SET_STENCIL_TEST_ENABLE
        g_state.rs[gpu::kRsStencilEnable] = v;
    } else if (m == 0x0210) {  // SET_SURFACE_COLOR_OFFSET: a target of the same size and format, until SetRenderTarget
        g_state.rt.data = v;
    } else {
        return false;
    }
    return true;
}

// the state a draw of the title's own arrays sees: the program from the start slot to its last instruction, each
// enabled array a stream of its own read from register r
// the program in the title's program memory (SelectVertexShader or a pushed start), over the API's streams laid out by
// the declaration of the handle last selected or set: what the library sends when the streams were set after the
// title's own arrays
// SelectVertexShaderDirect(attributes, slot) copies 16 attributes into a vertex shader of the library's own and
// selects that: per register its stream, offset in the vertex and D3DVSDT type, 16 bytes each
constexpr uint32_t kDirectAttributes = 0x14, kAttributeBytes = 16, kTypeNone = 0x02;
bool DirectHandle(uint32_t h) { return (h & 1) && !g_vs.count(h); }
void DirectInputs(uint32_t h, gpu::VertexShader& vs) {
    for (uint32_t r = 0; r < 16; r++) {
        uint32_t a = (h & ~1u) + kDirectAttributes + r * kAttributeBytes;
        uint32_t stream = R32(a), offset = R32(a + 4), type = R32(a + 8) & 0xFF;
        vs.inputs[r] = {};
        if (type != kTypeNone && stream < 16) vs.inputs[r] = {int(stream), offset, type};
    }
    for (uint32_t n = 0; n < 16; n++) vs.stream_stride[n] = g_state.streams[n].stride;
}

gpu::State SelectedProgramState() {
    gpu::State s = g_state;
    auto it = g_vs.find(g_raw.selected);
    bool direct = DirectHandle(g_raw.selected);
    if (it == g_vs.end() && !direct) return s;
    std::vector<uint32_t> program;
    for (uint32_t i = g_raw.start; i < 136; i++) {
        program.insert(program.end(), &g_raw.program[i * 4], &g_raw.program[i * 4 + 4]);
        if (g_raw.program[i * 4 + 3] & 1) break;  // the final instruction
    }
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t w) { h = (h ^ w) * 1099511628211ull; };
    for (uint32_t w : program) mix(w);
    mix(g_raw.selected);
    gpu::VertexShader layout;
    if (direct) {
        DirectInputs(g_raw.selected, layout);
        for (uint32_t r = 0; r < 16; r++) mix(layout.inputs[r].stream), mix(layout.inputs[r].offset), mix(layout.inputs[r].type);
        for (uint32_t n = 0; n < 16; n++) mix(layout.stream_stride[n]);
    }
    // the render thread tells shaders apart by pointer: one object per program and declaration
    static std::unordered_map<uint64_t, std::shared_ptr<gpu::VertexShader>> made;
    auto& vs = made[h];
    if (!vs) {
        vs = std::make_shared<gpu::VertexShader>(direct ? layout : *it->second);
        vs->handle = 1;
        vs->program = program;
        vs->program_type = 0x2078;
        vs->constants.clear();
    }
    s.vs = vs;
    s.vs_handle = 1;
    return s;
}

// an API draw's vertex state: the title's own arrays and program, its program over the API's streams, or the API's
gpu::State DrawState() {
    if (g_raw.arrays) return RawDrawState();
    if (g_raw.own_program && (g_vs.count(g_raw.selected) || DirectHandle(g_raw.selected))) return SelectedProgramState();
    return g_state;
}

gpu::State RawDrawState() {
    gpu::State s = g_state;
    std::vector<uint32_t> program;
    for (uint32_t i = g_raw.start; i < 136; i++) {
        program.insert(program.end(), &g_raw.program[i * 4], &g_raw.program[i * 4 + 4]);
        if (g_raw.program[i * 4 + 3] & 1) break;  // the final instruction
    }
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint32_t w) { h = (h ^ w) * 1099511628211ull; };
    for (uint32_t w : program) mix(w);
    for (uint32_t r = 0; r < 16; r++) mix(g_raw.format[r] & 0xFF);
    // the render thread tells shaders apart by pointer: one object per program and layout
    static std::unordered_map<uint64_t, std::shared_ptr<gpu::VertexShader>> made;
    auto& vs = made[h];
    if (!vs) {
        vs = std::make_shared<gpu::VertexShader>();
        vs->handle = 1;
        vs->program = program;
        vs->program_type = 0x2078;
        for (uint32_t r = 0; r < 16; r++) {
            uint32_t f = g_raw.format[r];
            if (!((f >> 4) & 0xF)) continue;  // size 0: disabled
            vs->inputs[r] = {int(r), 0, f & 0xFF};
            vs->stream_stride[r] = f >> 8;
        }
    }
    s.vs = vs;
    s.vs_handle = 1;
    for (uint32_t r = 0; r < 16; r++) {
        if (!g_raw.headers[r]) {
            g_raw.headers[r] = mem::AllocSystem(20);
            std::memset(HostPtr(g_raw.headers[r]), 0, 20);
            W32(g_raw.headers[r], 1);  // a vertex buffer, one reference
        }
        W32(g_raw.headers[r] + 4, g_raw.offset[r] & 0x7FFFFFFF);  // bit 31 picks the second vertex DMA context
        s.streams[r] = {g_raw.headers[r], g_raw.format[r] >> 8};
    }
    s.base_vertex = 0;
    return s;
}

// XBE_PUSH_STATS=1: every 5 s, the hardware methods titles pushed themselves that Walk let pass
void PushStat(uint32_t m) {
    static const bool on = std::getenv("XBE_PUSH_STATS") != nullptr;
    if (!on) return;
    static std::map<uint32_t, uint32_t> counts;
    static auto last = std::chrono::steady_clock::now();
    counts[m]++;
    if (std::chrono::steady_clock::now() - last < std::chrono::seconds(5)) return;
    last = std::chrono::steady_clock::now();
    std::string out;
    char b[24];
    for (auto& [k, n] : counts) {
        std::snprintf(b, sizeof(b), " %X:%u", k, n);
        out += b;
    }
    LOGE("push stats:%s", out.c_str());
    counts.clear();
}

void Walk(uint32_t end) {
    if (!g_walk || end < g_walk || end > g_scratch_push + g_scratch_size) return;
    for (uint32_t p = g_walk; p + 4 <= end;) {
        // count in bits 18 to 28, bit 30 keeps the method for every value
        uint32_t head = R32(p), method = head & 0x1FFC, n = (head >> 18) & 0x7FF;
        p += 4;
        if ((head & 0xA0030003) || n * 4 > end - p) {  // anything but a method header
            PushStat(0xF0000000u | (head & 0xE0000003u));
            break;
        }
        for (uint32_t i = 0; i < n; i++, p += 4) {
            uint32_t m = head & 0x40000000 ? method : method + i * 4, v = R32(p);
            float f, *a = g_im_attr;
            bool last = true;  // the slot's last component
            std::memcpy(&f, &v, 4);
            ApplyVisibilityMethod(m, v);
            if (m == 0x17FC) {  // begin/end, 0 ends
                if (v) {
                    g_im_prim = v;
                    g_im_mask = 0;
                    g_im_vertices.clear();
                    g_im_methods.clear();
                } else {
                    EndPrimitive();
                }
                continue;
            } else if (m >= 0x1880 && m < 0x1900) {  // 2 floats, z = 0 and w = 1
                a += (m - 0x1880) / 8 * 4;
                last = (m - 0x1880) / 4 & 1;
                a[last] = f;
                a[2] = 0;
                a[3] = 1;
            } else if (m >= 0x1940 && m < 0x1980) {  // 4 bytes
                a += (m - 0x1940) / 4 * 4;
                for (int c = 0; c < 4; c++) a[c] = (v >> c * 8 & 0xFF) / 255.0f;
            } else if (m >= 0x1A00 && m < 0x1B00) {  // 4 floats
                a += (m - 0x1A00) / 16 * 4;
                last = ((m - 0x1A00) / 4 & 3) == 3;
                a[(m - 0x1A00) / 4 & 3] = f;
            } else if (m >= 0x1500 && m < 0x150C) {  // position, 3 floats and w = 1
                last = m == 0x1508;
                a[(m - 0x1500) / 4] = f;
                a[3] = 1;
            } else if (m >= 0x1518 && m < 0x1528) {  // position, 4 floats
                last = m == 0x1524;
                a[(m - 0x1518) / 4] = f;
            } else {
                if (!ApplyRawMethod(m, v)) PushStat(m);
                continue;
            }
            if (!g_im_prim) continue;
            g_im_methods.push_back(m);
            g_im_methods.push_back(v);
            g_im_mask |= 1u << (a - g_im_attr) / 4;
            if (a == g_im_attr && last) g_im_vertices.insert(g_im_vertices.end(), g_im_attr, g_im_attr + 64);
        }
    }
    g_walk = end;
}
}  // namespace
}  // namespace d3d

HLE_STDCALL(D3DDevice_DrawVerticesUP, 4) {
    PushStat(0xD0000003u);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    LogQuadSource("up", a[0], 0, a[1], a[2], a[3]);
    DrawUp(a[0], a[1], HostPtr(a[2]), a[3]);
    return 0;
}

// BeginPush(dwords) hands out room for raw hardware commands, EndPush(end) runs what the title wrote there:
// a primitive between two begin/end writes with its vertices inline, any other method as a simple state.
HLE_STDCALL(D3DDevice_BeginPush, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_push_begin = ScratchPush(a[0] * 4);
    return g_push_begin;
}

HLE_STDCALL(D3DDevice_EndPush, 1) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    constexpr uint32_t kBeginEnd = 0x17FC, kInlineArray = 0x1818;
    static std::vector<uint32_t> vertices;
    uint32_t prim = 0;
    for (uint32_t p = g_push_begin, end = a[0]; g_push_begin && p + 4 <= end;) {
        // count in bits 18 to 28, bit 30 keeps the method for every value
        uint32_t head = R32(p), method = head & 0x1FFC, n = (head >> 18) & 0x7FF;
        p += 4;
        if ((head & 0xA0030003) || n * 4 > end - p) {  // anything but a method header
            LOG_FIRST(8, "d3d: EndPush stops at command %08X, %u bytes before the end", head, end - p);
            break;
        }
        for (uint32_t i = 0; i < n; i++, p += 4) {
            uint32_t m = head & 0x40000000 ? method : method + i * 4, v = R32(p);
            ApplyVisibilityMethod(m, v);
            if (m == kInlineArray) {
                if (prim) vertices.push_back(v);
            } else if (m != kBeginEnd) {
                PushStat(0x10000 | m);
                if (m >> 2 < g_state.simple.size()) g_state.simple[m >> 2] = v;
            } else if (v) {
                prim = v;
                vertices.clear();
            } else if (prim) {
                uint32_t stride = VertexBytes(), bytes = uint32_t(vertices.size()) * 4;
                if (stride && bytes % stride == 0) DrawUp(prim, bytes / stride, vertices.data(), stride);
                else LOG_FIRST(8, "d3d: EndPush primitive %u dropped, %u bytes of vertices of %u", prim, bytes, stride);
                prim = 0;
            }
        }
    }
    g_push_begin = 0;
    ScratchPush();
    return a[0];
}

// (index, result out, timestamp out): a report still pending is flushed through, as the console's GPU would
// have reached it by the time a title spins on it
HLE_STDCALL(D3DDevice_GetVisibilityTestResult, 3) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    uint32_t dev = R32(kDevicePtr), block = R32(dev + XDK_CDevice_VisibilityReports + (a[0] >> 8) * 4);
    if (!block) return 0x88760828u;  // D3DERR_TESTINCOMPLETE
    uint32_t report = block + (a[0] & 0xFF) * 16;
    if (R32(report + 12) == 0xFFFFFFFFu && g_backend) g_backend->Flush();
    if (R32(report + 12) == 0xFFFFFFFFu) return 0x88760828u;
    if (a[1]) W32(a[1], R32(report + 8));
    if (a[2]) {
        W32(a[2], R32(report));
        W32(a[2] + 4, R32(report + 4));
    }
    return 0;
}

// RunPushBuffer(push buffer, fixups): a recorded command buffer runs against the state set through the API. The
// titles seen record indexed draws in them: BEGIN_END around ARRAY_ELEMENT16 (two 16 bit indices a dword) and
// ARRAY_ELEMENT32 (one index, an odd count's last), drawn here as DrawIndexedVertices would. Inline vertices and
// other methods go the way EndPush takes them.
HLE_STDCALL(D3DDevice_RunPushBuffer, 2) {
    PushStat(0xD0000004u);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    constexpr uint32_t kBeginEnd = 0x17FC, kElement16 = 0x1800, kElement32 = 0x1808, kInlineArray = 0x1818;
    static std::vector<uint16_t> indices;
    static std::vector<uint32_t> vertices;
    if (a[1]) LOG_FIRST(4, "d3d: RunPushBuffer fixups %08X are not applied", a[1]);
    uint32_t pb = a[0], data = R32(pb + 4), size = R32(pb + 12);
    if (int32_t(R32(pb)) < 0) LOG_FIRST(4, "d3d: RunPushBuffer of a copied buffer %08X", pb);
    Walk(Dev(kDevPush));
    SyncDeferred();
    uint32_t prim = 0;
    bool wide = false;
    for (uint32_t p = data, end = data + size; p + 4 <= end;) {
        uint32_t head = R32(p), method = head & 0x1FFC, n = (head >> 18) & 0x7FF;
        p += 4;
        if ((head & 0xA0030003) || n * 4 > end - p) {
            LOG_FIRST(8, "d3d: RunPushBuffer stops at command %08X, %u bytes before the end", head, end - p);
            break;
        }
        for (uint32_t i = 0; i < n; i++, p += 4) {
            uint32_t m = head & 0x40000000 ? method : method + i * 4, v = R32(p);
            if (m == kElement16 || (m > kElement16 && m < kElement32 && !(head & 0x40000000))) {
                indices.push_back(uint16_t(v));
                indices.push_back(uint16_t(v >> 16));
            } else if (m == kElement32) {
                if (v > 0xFFFF) wide = true;
                indices.push_back(uint16_t(v));
            } else if (m == kInlineArray) {
                if (prim) vertices.push_back(v);
            } else if (m != kBeginEnd) {
                ApplyVisibilityMethod(m, v);
                if (!ApplyRawMethod(m, v) && m >> 2 < g_state.simple.size()) g_state.simple[m >> 2] = v;
            } else if (v) {
                prim = v;
                indices.clear();
                vertices.clear();
            } else if (prim) {
                if (wide) LOG_FIRST(4, "d3d: RunPushBuffer index above 65535 cut to 16 bits");
                if (!indices.empty() && g_backend)
                    g_backend->Draw(g_raw.arrays ? RawDrawState() : g_state, prim, 0, uint32_t(indices.size()),
                                    reinterpret_cast<const uint8_t*>(indices.data()));
                if (!vertices.empty()) {
                    uint32_t stride = VertexBytes(), bytes = uint32_t(vertices.size()) * 4;
                    if (stride && bytes % stride == 0) DrawUp(prim, bytes / stride, vertices.data(), stride);
                }
                prim = 0;
                wide = false;
            }
        }
    }
    return 0;
}

// frame buffer compression tiles are the console's memory controller's business
HLE_STDCALL(D3DDevice_SetTile, 2) { return 0; }

// the library's Begin pushes the start of the primitive, here it is opened
HLE_STDCALL(D3DDevice_Begin, 1) {
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Flags));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    g_im_prim = a[0];
    g_im_mask = 0;
    g_im_vertices.clear();
    SetDev(XDK_CDevice_Flags, Dev(XDK_CDevice_Flags) | 0x800);
    return 0;
}
// the library's End pushes the end of the primitive, here it is drawn
HLE_STDCALL(D3DDevice_End, 0) {
    PushStat(0xD0000005u);
    XDK_REQUIRE("d3d", XDK_KEY(CDevice_Flags));
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Walk(Dev(kDevPush));
    EndPrimitive();
    SetDev(XDK_CDevice_Flags, Dev(XDK_CDevice_Flags) & ~0x1800u);
    return 0;
}
