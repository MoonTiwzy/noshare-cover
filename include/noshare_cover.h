/*
 * noshare-cover: boundary between the Rust core and the Hyprland C++ shim.
 * Implemented in src/ffi.rs, src/extra.rs and src/hyprctl.rs. Change them together.
 *
 * All functions are called from the Hyprland render thread. None of them throws
 * or panics across the boundary. Frame data (pixels, fd) lives until nsc_end_frame().
 */
#ifndef NOSHARE_COVER_H
#define NOSHARE_COVER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* public API types; the shim doesn't need the binder, it is this plugin */
#define NOSHARE_COVER_NO_BINDER
#include "noshare_cover_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* path_cover; /* "" = default file from ~/.config/hypr */
    bool        loop;
    bool        no_file;    /* path_cover is an effect, not a file: don't look for the default one */
    double      speed;
    const char* backend;    /* "auto" | "gpu" | "cpu" */
    const char* gpu_device; /* "" = the compositor's GPU, or the first render node */
    uint64_t    render_dev; /* st_rdev of the DRM node the compositor draws with, 0 = unknown */
} nsc_settings;

/* Overrides from a window rule (NULL = field not set), and the size the cover is drawn
 * at, in capture pixels (0 = at its own size): a video shown smaller than it is gets
 * decoded into smaller frames. */
typedef struct {
    const char* rule_path;
    const char* rule_speed;
    const char* rule_loop;
    uint32_t    width;
    uint32_t    height;
} nsc_play_request;

enum {
    NSC_FRAME_NONE   = 0,
    NSC_FRAME_CPU    = 1, /* pixels: premultiplied BGRA, fourcc = DRM_FORMAT_ARGB8888 */
    NSC_FRAME_DMABUF = 2, /* planes[]: do NOT close fd, the core owns it; EGL doesn't take ownership */
};

typedef struct {
    int32_t  fd;
    uint32_t offset;
    uint32_t stride;
} nsc_plane;

typedef struct {
    uint64_t       cover_id;   /* stable texture cache key */
    uint64_t       generation; /* changed: update the texture */
    uint32_t       kind;
    uint32_t       width;
    uint32_t       height;
    uint32_t       stride;
    const uint8_t* pixels;
    uint32_t       fourcc;
    uint64_t       modifier;
    uint32_t       plane_count;
    nsc_plane      planes[4];
    uint64_t       buffer_id; /* dmabuf: same number = same memory, import it once; 0 = unknown */
} nsc_frame;

/* Rect from another plugin (see noshare_cover_api.h), client already stripped. */
typedef struct {
    double   x, y, w, h, rounding;
    uint64_t window; /* window address to fill with its cover, 0 = none */
    uint32_t fill;   /* 0 = black, 1 = window cover */
} nsc_extra_rect;

/* One window for nsc_annotate_clients. */
typedef struct {
    uint64_t    address;
    const char* rule_path; /* the rule's no_screen_share_cover, NULL = none */
    uint32_t    kind;      /* what covers it: 0 media, 1 transparent, 2 blur, 3 pixelate */
    int32_t     amount;    /* blur or pixelate size */
    bool        hidden;    /* no_screen_share applies to it */
} nsc_window_info;

/*
 * Struct layout is pinned by numbers on both sides: static_assert here, the
 * ffi::tests::abi_layout test in Rust. If they drift, the build or the test fails,
 * not Hyprland at runtime. Numbers are for LP64 (x86_64 / aarch64 Linux).
 */
#if defined(__LP64__)
#ifdef __cplusplus
#define NSC_ASSERT static_assert
#else
#define NSC_ASSERT _Static_assert
#endif
NSC_ASSERT(sizeof(nsc_frame) == 120 && offsetof(nsc_frame, buffer_id) == 112, "nsc_frame layout");
NSC_ASSERT(offsetof(nsc_frame, pixels) == 32 && offsetof(nsc_frame, modifier) == 48 && offsetof(nsc_frame, planes) == 60, "nsc_frame offsets");
NSC_ASSERT(sizeof(nsc_plane) == 12, "nsc_plane layout");
NSC_ASSERT(sizeof(nsc_settings) == 48 && offsetof(nsc_settings, speed) == 16 && offsetof(nsc_settings, render_dev) == 40, "nsc_settings layout");
NSC_ASSERT(sizeof(nsc_play_request) == 32 && offsetof(nsc_play_request, width) == 24, "nsc_play_request layout");
NSC_ASSERT(sizeof(nsc_extra_rect) == 56 && offsetof(nsc_extra_rect, fill) == 48, "nsc_extra_rect layout");
NSC_ASSERT(sizeof(noshare_cover_rect) == 56 && offsetof(noshare_cover_rect, window) == 40, "noshare_cover_rect layout");
NSC_ASSERT(sizeof(nsc_window_info) == 32 && offsetof(nsc_window_info, hidden) == 24, "nsc_window_info layout");
#undef NSC_ASSERT
#endif

bool     nsc_init(void);
void     nsc_shutdown(void);
void     nsc_set_settings(const nsc_settings* settings);
uint64_t nsc_epoch(void);
void     nsc_begin_frame(void);
/* false: nothing to draw. out->cover_id is set even then when the cover exists and only
 * has no frame yet. */
bool     nsc_resolve(const nsc_play_request* request, nsc_frame* out);
/* Milliseconds until the cover has a new frame: 0 = now, -1 = its picture won't change.
 * Asking keeps a video decoding between two frames. */
int32_t  nsc_next_frame_ms(uint64_t cover_id);
void     nsc_prewarm(const nsc_play_request* request);
void     nsc_end_frame(void);
bool     nsc_animating(void);
/* A NSC_FRAME_DMABUF frame of this cover couldn't be imported: it sends pixels from now on. */
void     nsc_refuse_video_memory(uint64_t cover_id);
bool     nsc_cover_alive(uint64_t cover_id);
size_t   nsc_take_notification(char* buf, size_t cap);
size_t   nsc_extra_rects(int64_t monitor_id, nsc_extra_rect* out, size_t cap);
/* The answer of `hyprctl clients` or `activewindow` (text, len bytes) with noScreenShare and
 * noScreenShareCover for every window in it that windows has. Returns the length of the
 * result, written to out only if it fits in cap. 0 = failed, keep text. */
size_t   nsc_annotate_clients(const char* text, size_t len, bool json, const nsc_window_info* windows, size_t count, char* out, size_t cap);

/* Public ABI implementation (noshare_cover_api.h). It is exported by the
 * noshare_cover_* wrappers in shim/main.cpp: the whole Rust archive is linked
 * hidden (--exclude-libs) so it doesn't share symbols with other plugins. */
uint32_t nsc_api_api_version(void);
bool     nsc_api_set_gone_callback(uint64_t client, void (*cb)(void* user), void* user);
void     nsc_api_notify_gone(void);
uint64_t nsc_api_register_client(const char* name);
void     nsc_api_unregister_client(uint64_t client);
bool     nsc_api_set_rects(uint64_t client, int monitor_id, const noshare_cover_rect* rects, size_t count);
bool     nsc_api_clear_client_rects(uint64_t client);
void     nsc_api_clear_extra_rects(void);
void     nsc_api_add_extra_rect(int monitor_id, double x, double y, double w, double h, double rounding);

#ifdef __cplusplus
}
#endif

#endif
