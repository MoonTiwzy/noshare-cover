# noshare-cover 2.0 architecture

## Why not "pure Rust"

A Hyprland plugin is a `.so` tied to the C++ ABI of one Hyprland build. Drawing into a
screencopy frame needs a hook on the private method `Screenshare::CScreenshareFrame::renderMonitor`
plus render pass classes (`CTexPassElement`, `CRectPassElement`), window/layer rules
(`Desktop::Rule`) and config values (`Config::Values`). Hyprland has no C API for any of
that, and its C++ ABI can't be called from Rust reliably (`SP<>` templates, virtual classes,
header versions). So: **all logic lives in Rust, with a thin C++ layer only where the C++
ABI is unavoidable.** The layer (`shim/`, a file per job, see `shim/shim.hpp`) has no media
logic.

```
                    ┌──────────────── Hyprland (C++) ───────────────────┐
  CScreenshareFrame │ renderMonitor ──hook──▶ shim/capture.cpp          │
                    │                         │ config, window/layer     │
                    │                         │ rules, geometry,         │
                    │                         │ textures, render pass    │
                    └─────────────────────────┼──────────────────────────┘
                                              │ C ABI (include/noshare_cover.h)
                    ┌─────────────────────────▼──────── Rust (src/) ─────┐
                    │ ffi.rs      entry points, panic guards              │
                    │ registry.rs covers: cache, epoch, errors, eviction  │
                    │ config.rs   settings, rule overrides, backend, GPU  │
                    │ media/      still (png/jpg) · gif · video           │
                    │   video/    demux → decode → YUV→BGRA · worker      │
                    │ gpu.rs      render node selection                   │
                    │ frame.rs    frame: BGRA pixels or dmabuf            │
                    │ extra.rs    public API for other plugins            │
                    └────────────────────────────────────────────────────┘
```

The VA-API decoder is a separate small library, `vaapi-helper` (cros-codecs over libva).
Its bytes are embedded into the plugin at build time and it is loaded from a memfd with
dlopen the first time a GPU video is opened. libva/libgbm are dependencies of the helper
only: without libva there is no VA-API, but the plugin still loads and uses the CPU.

## A screencopy frame

1. Hyprland renders a screencopy frame → our hook → the original → `paintCovers`.
2. The layer hands the current config to the core (`nsc_set_settings`, also on every config
   reload) when a value changed, and once a second anyway (the default file may appear). If
   it changed, the core closes all sources and bumps the epoch; the layer then drops its
   textures. The default cover is opened right away so the first capture already has a frame.
3. For every window and layer with `no_screen_share`: geometry as Hyprland computes it,
   rule overrides → `nsc_resolve` → frame. A surface's rules are collected once per frame
   (`CRuleCache`), however often the frame asks for them. Covers with the same (file, speed,
   loop) share one source and one texture; a source is polled once per frame however many
   windows show it. Every request says how large the cover is drawn; the source gets the
   largest size asked for lately and may hand out smaller frames (see Video).
4. Frame → texture: pixels via `createTexture(drmFormat, pixels, stride, size)` /
   `ITexture::update`, dmabuf via `createTexture(SDMABUFAttrs)`, once per decoder buffer: the
   decoder puts its frames into a few buffers in turn (`buffer_id`), and a texture imported
   from one shows whatever was decoded into it last. The texture is only updated when
   `generation` changes. A dmabuf Hyprland can't import is refused
   (`nsc_refuse_video_memory`): the cover sends pixels from then on.
5. Rectangles from other plugins (gloview's overview tiles) get black or the window's cover.
6. `nsc_end_frame`: covers nobody showed for 30 s are closed (for video that stops the
   thread). Every 120 frames the layer drops textures of dead covers.
7. While a monitor is being shared and shows a media cover, a timer (the pump) asks the core
   when each cover on it has its next frame (`nsc_next_frame_ms`), damages the cover's box
   when the frame is there and sleeps until the next one is due: the monitor repaints at the
   cover's frame rate, and not at all for a still. Hyprland only produces capture frames when
   the monitor repaints, so a video cover would otherwise stall on a static monitor. The
   question also keeps a video decoding between two frames.
8. A capture into a client's dmabuf has no texture to read back, which blur under a
   translucent window over a cover needs: such a frame is drawn into a framebuffer of ours and
   copied over. Only such a frame: one without a hidden window and a translucent one on the
   monitor goes straight into the client's buffer.

## Video

One thread per source. Only the latest decoded frame is exposed (no queue, no memory
growth). The thread sleeps when nobody asks for frames for 400 ms; after waking up the
clock restarts from the current frame instead of catching up. Frames more than 80 ms late
are dropped. The first poll of a new source waits up to 120 ms for its first frame.
Dropping a source = `quit` + `notify` + `join`: unloading the plugin leaves no threads.
A decoder panic is caught in the thread and reported as an error.

Backend (`backend` in the config): `auto` tries the GPU (NVDEC on NVIDIA render nodes, then
VA-API) and falls back to the CPU; `gpu` never falls back; `cpu` is software only.
`gpu_device` pins a render node; otherwise it is the GPU the compositor draws with (the layer
passes its DRM device number), or the first `/dev/dri/renderD*` if that isn't known. Every
refusal is explained in a message the user sees.

A frame reaches the layer one of two ways.

As pixels: YUV → BGRA on the worker thread (`yuv::Converter`, one per decoder). Samples above
8 bit are cut to 8 first; on x86-64 with AVX2 rows are converted 8 pixels at a time with the
scalar code's integer math (identical output, about 0.6 ms for 1080p on one thread; without
AVX2 large frames are split across up to 4 threads). A video shown smaller than it is gets
halved (2x2 averages) as often as it still covers the size it is drawn at, so less is
converted and uploaded. Output buffers are reused, not allocated per frame, and given back
when nobody watches.

In video memory: VA-API on the compositor's own GPU. The helper has the GPU's video processor
(VPP) convert each decoded NV12 frame into an RGB surface of the size it is drawn at and
exports that as a dma-buf; the layer imports it. Nothing is copied to system memory. The
picture belongs to its frame until the frame is dropped, then its token goes back to the
helper with the next packet. The colours are given to the VPP explicitly (matrix and range
only, same primaries and transfer on both sides), so the result matches the pixel path.
NVDEC keeps copying: its frames can't be exported as dma-bufs.

## Coexisting with other plugins

Hyprland's function hooks are exclusive: two plugins can't hook `renderMonitor` at once.
gloview hooks it itself while noshare-cover isn't loaded. If the hook is taken when
noshare-cover loads, noshare-cover keeps retrying instead of refusing to load; gloview
re-checks after every config reload (Hyprland reloads after each plugin load/unload),
lets go of the hook and switches to noshare-cover's API. On unload noshare-cover removes
its hook first and then calls API clients' gone callbacks, so gloview can take the hook
back and never has to keep a dlopen handle that would pin noshare-cover in memory.

## Lifecycle and safety

- No panic reaches Hyprland: every `extern "C"` entry is wrapped in `catch_unwind`.
- `PLUGIN_EXIT`: stop timers, remove the hook first (Hyprland removes hooks only *after*
  EXIT), notify API clients, drop textures, `nsc_shutdown` (joins all threads), then rule
  effects and config values.
- The Rust archive is linked hidden (`--exclude-libs,ALL`); the public API is exported by
  thin wrappers in `shim/main.cpp`. Do not use a version script with `local: *`: it also
  localizes Hyprland's inline globals (`g_pHyprRenderer` & co) and the plugin ends up with
  its own null copies.
- `-fno-gnu-unique`: without it glibc marks the `.so` NODELETE and a reload silently gets
  the old image.
- On load the plugin compares `__hyprland_api_get_hash()` with the headers it was built
  against and refuses to load on mismatch instead of crashing the compositor.
- C ABI struct layouts are pinned by numbers on both sides (`static_assert` in the header,
  the `ffi::tests::abi_layout` test).

## Status

| Part | State |
|---|---|
| Config, window and layer rules (flat Lua fields), `backend`, `gpu_device` | ✅ tests + live Hyprland |
| PNG / JPEG / GIF | ✅ tests + live Hyprland |
| MP4/MOV, WebM/MKV demuxers | ✅ tests on real files |
| CPU: AV1 (rav1d), H.264 (openh264), VP8/VP9 (libvpx), YUV→BGRA (AVX2 and scalar, halving) | ✅ tests on real files + live Hyprland |
| NVDEC (dlopen, driver parser, NV12/P016 copy) | ⚠️ builds, struct layouts checked against ffnvcodec-headers; not run on a GPU |
| VA-API (vaapi-helper on cros-codecs, embedded, loaded from memfd) | ✅ H.264, VP9, AV1 on an AMD iGPU (radeonsi), tests with `NSC_TEST_GPU`; 8-bit only |
| VA-API frames in video memory (VPP → RGB dma-buf → Hyprland texture) | ✅ radeonsi: `NSC_TEST_GPU` tests against the CPU's picture, `tests/e2e/video-memory.sh` in a Hyprland on that GPU; ⚠️ not run on Intel |
| Repaint pump by frame time | ✅ `tests/e2e/pump.sh`: 10 and 30 fps covers, a still |
| `shim/` (0.57.x) | ✅ builds and runs on 0.57.0 |
| Together with gloview | ✅ `tests/e2e/with-gloview.sh`: both load orders, unload/reload, overview tiles |
| Unload/load, leaks | ✅ `tests/e2e/run.sh`: 8 cycles, threads and RSS stable |
| Packaging: hyprpm, Arch PKGBUILD, Nix flake | ✅ makepkg; nix build against 0.57.0 in CI; hyprpm pins 0.56.x to the last version that supports it |

## What to check on a real machine

- NVDEC on NVIDIA and VA-API on Intel/AMD (`backend = "gpu"`, `NOSHARE_COVER_DEBUG`);
- VA-API frames in video memory on Intel (iHD): `tests/e2e/video-memory.sh`, and the trace
  line `from video memory ... imported: ok`;
- long video runs (hours): Hyprland RSS and fd count stay flat;
- two GPUs with `gpu_device`;
- portal-based capture (OBS, Discord) on a static monitor while working on another one.
