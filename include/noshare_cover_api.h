/*
 * noshare-cover: public API for other Hyprland plugins.
 *
 * Why: a window overlay plugin (e.g. gloview) draws previews of no_screen_share
 * windows into its own frame. noshare-cover only hides the window itself in the
 * screencast, so the preview would leak. Through this API a plugin says "cover
 * this area too", either in black or with that window's cover.
 *
 * Usage: include this header only, nothing to link:
 *
 *     #include "noshare_cover_api.h"
 *
 *     static noshare_cover_api g_nsc;          // in PLUGIN_INIT:
 *     if (noshare_cover_bind(&g_nsc) == 0)
 *         g_client = g_nsc.register_client("gloview");
 *
 *     // every overlay frame (or when tiles move), per monitor:
 *     noshare_cover_rect r[] = {{x, y, w, h, rounding, window_address, NOSHARE_COVER_FILL_WINDOW}};
 *     g_nsc.set_rects(g_client, monitor_id, r, 1);
 *
 *     // overlay closed:             g_nsc.set_rects(g_client, monitor_id, NULL, 0);
 *     // unloading your plugin:      g_nsc.unregister_client(g_client); noshare_cover_unbind(&g_nsc);
 *
 * Coordinates are global logical layout pixels (same space as window position
 * and size). noshare-cover subtracts the monitor position, multiplies by scale
 * and accounts for the capture region itself. Rects persist until replaced.
 *
 * Why bind via dl_iterate_phdr: Hyprland loads plugins with RTLD_LOCAL, so
 * dlsym(RTLD_DEFAULT, ...) can't see their symbols.
 */
#ifndef NOSHARE_COVER_API_H
#define NOSHARE_COVER_API_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define NOSHARE_COVER_API_VERSION 2

enum {
    NOSHARE_COVER_FILL_BLACK  = 0,
    /* Cover of `window` (per its rules or the global one). No window/cover: black. */
    NOSHARE_COVER_FILL_WINDOW = 1,
};

typedef struct {
    double   x, y, w, h;
    double   rounding;
    uint64_t window; /* window address, as `address` in `hyprctl clients -j`; 0 = none */
    uint32_t fill;   /* NOSHARE_COVER_FILL_* */
} noshare_cover_rect;

typedef struct {
    uint32_t version; /* as reported by the plugin */
    void*    handle;  /* RTLD_NOLOAD reference, closed in noshare_cover_unbind */

    /* v2 */
    uint64_t (*register_client)(const char* name);
    void (*unregister_client)(uint64_t client);
    bool (*set_rects)(uint64_t client, int monitor_id, const noshare_cover_rect* rects, size_t count);
    bool (*clear_client_rects)(uint64_t client);

    /* v2, optional (NULL in older builds): "going away". The callback fires when
     * noshare-cover unloads, after its renderMonitor hook is removed: drop all pointers
     * from here, renderMonitor is free. With a callback set you can release the handle
     * right away (noshare_cover_drop_handle) so it doesn't block noshare-cover unload. */
    bool (*set_gone_callback)(uint64_t client, void (*cb)(void* user), void* user);

    /* v1, kept for compatibility */
    void (*clear_extra_rects)(void);
    void (*add_extra_rect)(int monitor_id, double x, double y, double w, double h, double rounding);
} noshare_cover_api;

#if defined(__linux__) && !defined(NOSHARE_COVER_NO_BINDER)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <link.h>
#include <string.h>

static int noshare_cover__find(struct dl_phdr_info* info, size_t size, void* data) {
    (void)size;
    if (!info->dlpi_name || !strstr(info->dlpi_name, "noshare-cover"))
        return 0;
    *(const char**)data = info->dlpi_name;
    return 1;
}

/*
 * Find the loaded noshare-cover and bind its functions.
 * 0: ok (v2 available); 1: plugin not loaded; 2: old version without v2
 * (v1 functions are still filled in if present); 3: dlopen failed.
 */
static inline int noshare_cover_bind(noshare_cover_api* api) {
    memset(api, 0, sizeof *api);

    const char* path = NULL;
    dl_iterate_phdr(noshare_cover__find, &path);
    if (!path)
        return 1;

    void* h = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
    if (!h)
        return 3;
    api->handle = h;

#define NOSHARE_COVER__SYM(field, name) *(void**)(&api->field) = dlsym(h, name)
    NOSHARE_COVER__SYM(clear_extra_rects, "noshare_cover_clear_extra_rects");
    NOSHARE_COVER__SYM(add_extra_rect, "noshare_cover_add_extra_rect");

    uint32_t (*version)(void) = NULL;
    *(void**)(&version)       = dlsym(h, "noshare_cover_api_version");
    api->version              = version ? version() : 1;
    if (api->version < 2)
        return 2;

    NOSHARE_COVER__SYM(register_client, "noshare_cover_register_client");
    NOSHARE_COVER__SYM(unregister_client, "noshare_cover_unregister_client");
    NOSHARE_COVER__SYM(set_rects, "noshare_cover_set_rects");
    NOSHARE_COVER__SYM(clear_client_rects, "noshare_cover_clear_client_rects");
    NOSHARE_COVER__SYM(set_gone_callback, "noshare_cover_set_gone_callback");
#undef NOSHARE_COVER__SYM

    return (api->register_client && api->unregister_client && api->set_rects && api->clear_client_rects) ? 0 : 2;
}

/* Release the RTLD_NOLOAD reference but keep the pointers. Only together with
 * set_gone_callback: without it the pointers dangle after noshare-cover unloads. Don't
 * hold the handle longer than needed: while it's open, dlclose can't unload noshare-cover. */
static inline void noshare_cover_drop_handle(noshare_cover_api* api) {
    if (api->handle)
        dlclose(api->handle);
    api->handle = NULL;
}

static inline void noshare_cover_unbind(noshare_cover_api* api) {
    if (api->handle)
        dlclose(api->handle);
    memset(api, 0, sizeof *api);
}

#endif /* __linux__ */

#endif
