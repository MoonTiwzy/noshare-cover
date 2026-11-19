// Covers in the capture frame: media, blur and pixelate, black where nothing else fits, and
// what is stacked above a hidden surface drawn again over its cover.

#include "shim.hpp"

namespace nsc {

    namespace {

        // Texture cache keyed by cover id. Cleared entirely when the core epoch changes.
        struct SCachedTexture {
            SP<Render::ITexture> tex;
            uint64_t             generation = 0;
            uint32_t             w = 0, h = 0;
            bool                 dmabuf = false;
            // Frames in video memory: the decoder puts them into a few buffers in turn, and a
            // texture imported from a buffer shows whatever was last decoded into it. So each
            // buffer is imported once (by its id) and a new frame only picks its texture.
            std::unordered_map<uint64_t, SP<Render::ITexture>> buffers;
        };

        // more than a decoder ever has at a time: beyond it the ids are of buffers long gone
        constexpr size_t MAX_BUFFER_TEXTURES = 32;

        std::unordered_map<uint64_t, SCachedTexture> g_textures;

        uint64_t                                     g_epoch      = 0;

        uint64_t                                     g_frameCount = 0;

        // Core frame -> Hyprland texture. A new texture only when the frame changed.
        SP<Render::ITexture> textureFor(const nsc_frame& f) {
            if (const auto epoch = nsc_epoch(); epoch != g_epoch) { // the core reset its covers
                g_textures.clear();
                g_epoch = epoch;
            }
            auto& c = g_textures[f.cover_id];
            if (c.tex && c.generation == f.generation)
                return c.tex;

            if (f.kind == NSC_FRAME_CPU && f.pixels) {
                c.buffers.clear();
                auto* px = const_cast<uint8_t*>(f.pixels);
                if (c.tex && !c.dmabuf && c.w == f.width && c.h == f.height && c.tex->ok()) {
                    const CRegion damage{0.0, 0.0, double(f.width), double(f.height)};
                    c.tex->update(f.fourcc, px, f.stride, damage);
                } else {
                    c.tex = g_pHyprRenderer->createTexture(f.fourcc, px, f.stride, Vector2D{double(f.width), double(f.height)});
                }
                c.dmabuf = false;
            } else if (f.kind == NSC_FRAME_DMABUF && f.plane_count > 0 && f.plane_count <= 4) {
                Aquamarine::SDMABUFAttrs attrs;
                attrs.success  = true;
                attrs.size     = Vector2D{double(f.width), double(f.height)};
                attrs.format   = f.fourcc;
                attrs.modifier = f.modifier;
                attrs.planes   = int(f.plane_count);
                for (uint32_t i = 0; i < f.plane_count; ++i) {
                    attrs.fds[i]     = f.planes[i].fd;
                    attrs.offsets[i] = f.planes[i].offset;
                    attrs.strides[i] = f.planes[i].stride;
                }
                if (c.w != f.width || c.h != f.height)
                    c.buffers.clear(); // the decoder changed the size: those buffers are gone
                auto known = f.buffer_id ? c.buffers.find(f.buffer_id) : c.buffers.end();
                if (known != c.buffers.end() && known->second && known->second->ok() && known->second->m_size == attrs.size)
                    c.tex = known->second;
                else {
                    c.tex = g_pHyprRenderer->createTexture(attrs);
                    NSC_TRACE("cover %llu: %ux%u from video memory, buffer %llu imported: %s\n", static_cast<unsigned long long>(f.cover_id), f.width, f.height,
                              static_cast<unsigned long long>(f.buffer_id), c.tex && c.tex->ok() ? "ok" : "failed");
                    if (!c.tex || !c.tex->ok()) {
                        // this GPU's driver doesn't take what the decoder exported: black for
                        // this frame, and the core sends pixels from the next one on
                        nsc_refuse_video_memory(f.cover_id);
                        c = SCachedTexture{};
                        return nullptr;
                    }
                    if (f.buffer_id) {
                        if (c.buffers.size() >= MAX_BUFFER_TEXTURES)
                            c.buffers.clear();
                        c.buffers[f.buffer_id] = c.tex;
                    }
                }
                c.dmabuf = true;
            } else {
                return nullptr;
            }

            c.w          = f.width;
            c.h          = f.height;
            c.generation = f.generation;
            return (c.tex && c.tex->ok()) ? c.tex : nullptr;
        }

        // clip: where drawing is allowed (the draw is scissored to it); empty = the whole box
        void drawBlack(Render::CRenderContext& ctx, const CBox& box, int round = 0, float roundingPower = 2.F, const CRegion& clip = {}) {
            const CRegion area = clip.empty() ? CRegion{box} : clip;
            g_painted.add(area);
            g_pHyprRenderer->draw(ctx, CRectPassElement::SRectData{.box = box, .color = CHyprColor{0.F, 0.F, 0.F, 1.F}, .round = round, .roundingPower = roundingPower}, area);
        }

        // drawEffect's intermediate buffers by size, per monitor, dropped after a while unused
        // (fxBuffer, pruneTextures)
        struct SFxBuffer {
            SP<Render::IFramebuffer> fb;
            uint64_t                 used        = 0; // g_frameCount then
            uintptr_t                stream      = 0; // the capture stream that used it last
            uint64_t                 streamFrame = 0; // that stream's frame then
        };

        // Capture streams (a client's session on a monitor) take turns, at their own rates
        // and each with the hidden windows it sees as they are (revealFor), so each needs
        // sizes the others may not: a buffer ages only by the frames of the stream that used
        // it last, never by another monitor's or another client's.
        struct SFxStream {
            uint64_t frames = 0; // its capture frames so far
            uint64_t seen   = 0; // g_frameCount at its last one
        };

        struct SMonitorFx {
            std::unordered_map<uintptr_t, SFxStream> streams; // by session (see paintCovers)
            std::map<std::pair<int, int>, SFxBuffer> buffers;
        };

        std::unordered_map<MONITORID, SMonitorFx> g_fx;
        uintptr_t                                 g_fxStream = 0; // the stream being drawn

        SP<Render::IFramebuffer> fxBuffer(const PHLMONITOR& mon, const Vector2D& size, const SP<Render::ITexture>& like) {
            auto& fx = g_fx[mon ? mon->m_id : MONITOR_INVALID];
            // a box that changes size every frame (an animation, a resize) asks for new sizes
            // every frame: don't keep those the stream that used them last has drawn a frame
            // without since
            if (fx.buffers.size() > 32)
                std::erase_if(fx.buffers, [&](const auto& kv) {
                    const auto s = fx.streams.find(kv.second.stream);
                    return s == fx.streams.end() || kv.second.streamFrame + 1 < s->second.frames;
                });
            auto& b = fx.buffers[{int(size.x), int(size.y)}];
            if (!b.fb)
                b.fb = g_pHyprRenderer->createFB("noshare-cover effect");
            if (!b.fb || !b.fb->alloc(int(size.x), int(size.y)))
                return nullptr;
            b.fb->setImageDescription(like->m_imageDescription);
            b.used        = g_frameCount;
            b.stream      = g_fxStream;
            b.streamFrame = fx.streams[g_fxStream].frames;
            return b.fb;
        }

        // Blur or pixelate what the frame shows at box, taken from the image Hyprland draws the
        // frame from (see renderMonitorWithCovers), and draw it back there, only within clip.
        // The picture is halved with linear filtering, which averages 2x2 pixels, until one
        // pixel stands for `amount`; pixelate scales that up with nearest neighbour (blocks),
        // blur goes back up through the same sizes with linear filtering (smooth).
        bool drawEffect(Render::CRenderContext& ctx, const SP<Render::ITexture>& image, const SCoverKind& kind, const CBox& box, const Vector2D& capturePos, int round, float roundingPower,
                        const CRegion& clip) {
            if (!image || box.w < 1 || box.h < 1 || image->m_size.x < 1 || image->m_size.y < 1)
                return false;
            const CBox src = box.copy().translate(capturePos); // in the image
            // Never a buffer larger than the image, which the GPU has already taken: a zoomed box
            // is several times the screen, and a buffer past the GPU's texture limit takes the
            // compositor down (Hyprland asserts on an incomplete framebuffer). A box that large
            // gets larger blocks than asked for, never smaller ones.
            const Vector2D target{std::clamp(std::round(src.w / kind.amount), 1.0, std::floor(image->m_size.x)),
                                  std::clamp(std::round(src.h / kind.amount), 1.0, std::floor(image->m_size.y))};
            // first to target * 2^k, the nearest such size to src, so that every halving after it
            // is exact: one pixel from 2x2
            int k = std::clamp(int(std::lround(std::log2(std::max(src.w / target.x, src.h / target.y)))), 0, 12);
            // the same for the sizes it is halved from
            while (k > 0 && (target.x * (1 << k) > image->m_size.x || target.y * (1 << k) > image->m_size.y))
                --k;
            std::vector<Vector2D> steps;
            for (int i = k; i >= 0; --i)
                steps.push_back(target * double(1 << i));
            if (kind.kind == SCoverKind::BLUR)
                for (int i = 1; i <= k; ++i)
                    steps.push_back(target * double(1 << i));

            SP<Render::ITexture> tex  = image;
            CBox                 from = src; // the part of tex that is wanted
            const auto           mon  = ctx.m_data.pMonitor.lock();
            {
                const auto                          viewport = ctx.m_data.fbSize;
                const Hyprutils::Utils::CScopeGuard restoreViewport{[&] { g_pHyprRenderer->setViewport(0, 0, viewport.x, viewport.y); }};
                const auto                          state = ctx.saveDrawState();
                ctx.m_data.renderModif.enabled            = false;
                ctx.m_data.useNearestNeighbor             = false;
                for (const auto& size : steps) {
                    const auto fb = fxBuffer(mon, size, image);
                    if (!fb)
                        return false;
                    const auto    bound = g_pHyprRenderer->bindTempFB(ctx, fb);
                    const CRegion all{CBox{{}, size}};
                    ctx.m_data.fbSize = size;
                    g_pHyprRenderer->setProjectionType(ctx, Render::RPT_EXPORT);
                    g_pHyprRenderer->setViewport(0, 0, size.x, size.y);
                    const double sx = size.x / from.w, sy = size.y / from.h;
                    g_pHyprRenderer->draw(ctx, CClearPassElement::SClearData{{0.F, 0.F, 0.F, 1.F}}, all); // beyond the image (off screen)
                    g_pHyprRenderer->draw(ctx, CTexPassElement::SRenderData{.tex = tex, .box = {-from.x * sx, -from.y * sy, tex->m_size.x * sx, tex->m_size.y * sy}}, all);
                    tex  = fb->getTexture();
                    from = CBox{{}, size};
                }
            }
            const bool nearest            = std::exchange(ctx.m_data.useNearestNeighbor, kind.kind == SCoverKind::PIXELATE);
            const Hyprutils::Utils::CScopeGuard restoreFilter{[&] { ctx.m_data.useNearestNeighbor = nearest; }};
            g_pHyprRenderer->draw(ctx, CTexPassElement::SRenderData{.tex = tex, .box = box, .round = round, .roundingPower = roundingPower}, clip);
            return true;
        }

        // Popups (menus, tooltips) of a hidden window or layer: blurred or pixelated like it,
        // otherwise black like Hyprland does.
        template <class V>
        void paintPopupBoxes(Render::CRenderContext& ctx, const V& view, const Vector2D& baseOffset, const PHLMONITOR& mon, const Vector2D& capturePos, const SP<Render::ITexture>& image,
                             const SCoverKind& kind) {
            const auto head = view->popupHead();
            if (!head)
                return;
            const bool effect = kind.kind == SCoverKind::BLUR || kind.kind == SCoverKind::PIXELATE;
            head->breadthfirst(
                [&](SP<Desktop::View::CPopup> popup, void*) {
                    if (!popup || !popup->wlSurface() || !popup->wlSurface()->resource() || !viewVisible(popup))
                        return;
                    const auto rel = popup->coordsRelativeToParent();
                    popup->wlSurface()->resource()->breadthfirst(
                        [&](SP<CWLSurfaceResource> surf, const Vector2D& local, void*) {
                            const auto box = zoomed(mon, CBox{baseOffset + rel + local, surf->m_current.size}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
                            if (box.w < 1 || box.h < 1)
                                return;
                            if (effect && drawEffect(ctx, image, kind, box, capturePos, 0, 2.F, CRegion{box}))
                                g_painted.add(box);
                            else
                                drawBlack(ctx, box);
                        },
                        nullptr);
                },
                nullptr);
        }

        // Window cover for another plugin's rect: same rules as for the window itself.
        // size: of the rect, in capture pixels. cover: the core's id of the media it shows or
        // waits for (0: none), for the pump.
        SP<Render::ITexture> coverForWindowAddress(uint64_t address, const Vector2D& size, uint64_t& cover) {
            cover = 0;
            if (!address)
                return nullptr;
            for (const auto& w : Desktop::windowState()->windows()) {
                if (!w || reinterpret_cast<uintptr_t>(w.get()) != address)
                    continue;
                const auto rules = ruleValuesFor(w);
                if (coverKind(rules).kind != SCoverKind::MEDIA)
                    return nullptr; // another plugin's picture of it can only be blacked out
                const auto req   = playRequest(rules, size);
                nsc_frame  f{};
                const bool ready = nsc_resolve(&req, &f);
                cover            = f.cover_id; // also while it is still being opened
                return ready ? textureFor(f) : nullptr;
            }
            return nullptr;
        }

        // The part of a surface's buffer the client shows (its wp_viewporter source), as texture
        // UVs, computed like Hyprland's calculateUVForSurface; nullopt for the whole buffer.
        std::optional<std::pair<Vector2D, Vector2D>> viewportUV(const SP<CWLSurfaceResource>& surf) {
            const auto& st = surf->m_current;
            if (!st.viewport.hasSource || st.bufferSize.x <= 0 || st.bufferSize.y <= 0)
                return std::nullopt;
            // viewporter order: buffer transform, then buffer scale, then the source crop
            const Vector2D size = st.transform % 2 == 1 ? Vector2D{st.bufferSize.y, st.bufferSize.x} : st.bufferSize;
            CBox           src  = st.viewport.source;
            src.scale(st.scale);
            src.transform(Math::wlTransformToHyprutils(Math::invertTransform(st.transform)), size.x, size.y);
            const Vector2D topLeft{src.x / st.bufferSize.x, src.y / st.bufferSize.y};
            const Vector2D bottomRight{(src.x + src.width) / st.bufferSize.x, (src.y + src.height) / st.bufferSize.y};
            if (bottomRight.x < 0.00001 || bottomRight.y < 0.00001)
                return std::nullopt;
            return std::pair{topLeft, bottomRight};
        }

        // Draw a client surface's texture into box (its logical size), cropped to what the
        // client shows. Chromium keeps an oversized buffer and shows a viewporter crop of it:
        // drawn whole, the page shrank into a corner of the window and blur filled the rest.
        void drawSurfaceTexture(Render::CRenderContext& ctx, const SP<CWLSurfaceResource>& surf, CTexPassElement::SRenderData data, const CRegion& damage) {
            const auto uv = viewportUV(surf);
            if (!uv) {
                g_pHyprRenderer->draw(ctx, data, damage);
                return;
            }
            auto&                         rd          = ctx.m_data;
            const auto                    topLeft     = rd.primarySurfaceUVTopLeft;
            const auto                    bottomRight = rd.primarySurfaceUVBottomRight;
            Hyprutils::Utils::CScopeGuard restore{[&] {
                rd.primarySurfaceUVTopLeft     = topLeft;
                rd.primarySurfaceUVBottomRight = bottomRight;
            }};
            rd.primarySurfaceUVTopLeft     = uv->first;
            rd.primarySurfaceUVBottomRight = uv->second;
            data.allowCustomUV             = true;
            g_pHyprRenderer->draw(ctx, data, damage);
        }

        // Draw a layer again from its surface textures with its alpha, limited to clip.
        void redrawLayer(Render::CRenderContext& ctx, const PHLLS& l, const CRegion& clip, const PHLMONITOR& mon, const Vector2D& capturePos) {
            const auto res = l->resource();
            if (!res)
                return;
            float alpha = 1.F;
            for (uint8_t k = 0; k < Desktop::View::LS_ALPHA_LAST; ++k)
                alpha *= l->alpha().get(k)->value();
            const auto base = l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
            res->breadthfirst(
                [&](SP<CWLSurfaceResource> surf, const Vector2D& local, void*) {
                    const auto tex = surf->m_current.texture;
                    if (!tex)
                        return;
                    const auto box = zoomed(mon, CBox{base + local, surf->m_current.size}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
                    if (box.w >= 1 && box.h >= 1)
                        drawSurfaceTexture(ctx, surf, CTexPassElement::SRenderData{.tex = tex, .box = box, .a = alpha, .clipRegion = clip}, clip);
                },
                nullptr);
        }

        // Every couple of seconds, drop textures of covers the core has already closed, and
        // effect buffers that weren't used in that time.
        void pruneTextures() {
            if (++g_frameCount % 120 != 0)
                return;
            std::erase_if(g_textures, [](const auto& kv) { return !nsc_cover_alive(kv.first); });
            for (auto& [id, fx] : g_fx) {
                std::erase_if(fx.buffers, [](const auto& kv) { return kv.second.used + 120 < g_frameCount; });
                std::erase_if(fx.streams, [](const auto& kv) { return kv.second.seen + 120 < g_frameCount; });
            }
            std::erase_if(g_fx, [](const auto& kv) { return kv.second.buffers.empty() && kv.second.streams.empty(); });
        }

        // Rects from other plugins (noshare_cover_api.h): an overview's tiles of hidden windows.
        // revealed: hidden windows this client sees as they are (see revealFor), tiles too.
        void paintExtraRects(Render::CRenderContext& ctx, const PHLMONITOR& mon, const Vector2D& capturePos, const std::unordered_set<uintptr_t>& revealed) {
            // Usually a few dozen overlay tiles; if there are more, take as many as reported.
            std::vector<nsc_extra_rect> rects(64);
            size_t                      n = nsc_extra_rects(mon->m_id, rects.data(), rects.size());
            if (n > rects.size()) {
                rects.resize(n);
                n = std::min(nsc_extra_rects(mon->m_id, rects.data(), rects.size()), rects.size());
            }

            for (size_t i = 0; i < n; ++i) {
                const auto& r = rects[i];
                if (r.window && revealed.contains(uintptr_t(r.window)))
                    continue;
                const CBox global{r.x, r.y, std::max(r.w, 1.0), std::max(r.h, 1.0)};
                const auto box   = zoomed(mon, global.copy().translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
                const int  round = int(std::lround(r.rounding * mon->m_scale * zoomScale(mon)));
                if (box.w < 1 || box.h < 1)
                    continue;

                if (r.fill == 1) {
                    uint64_t   cover = 0;
                    const auto tex   = coverForWindowAddress(r.window, box.size(), cover);
                    // a video plays on the tile, and a cover still being opened shows up there,
                    // also on a still screen: the pump repaints the tile as for the window
                    if (cover)
                        g_animBoxes.push_back({mon, global, cover, g_fxStream});
                    if (tex) {
                        drawBlack(ctx, box, round); // under the cover's transparent pixels
                        g_pHyprRenderer->draw(ctx, CTexPassElement::SRenderData{.tex = tex, .box = box, .round = round}, box);
                        continue;
                    }
                }
                drawBlack(ctx, box, round);
            }
        }

        // Take an opaque box out of a region, but keep its r x r corners: a rounded window
        // is transparent there, and whatever is below would show through. The window is
        // drawn again over its corners afterwards (see paintCovers).
        void subtractRounded(CRegion& region, const CBox& b, int r) {
            r = std::clamp(r, 0, int(std::min(b.w, b.h) / 2));
            region.subtract(CBox{b.x + r, b.y, b.w - 2 * r, b.h});
            region.subtract(CBox{b.x, b.y + r, b.w, b.h - 2 * r});
        }

        // Draw a window again from its surface textures, with its alpha, rounding and blur,
        // limited to clip. Used for windows above a hidden one: the cover went over them, and
        // this puts them back on top of it.
        // canvas: draw it as it stands still into a buffer that starts at this monitor pixel
        // (see drawBent), without blur: there is nothing under it to blur there. Nor under an
        // opaque window, as in Hyprland.
        void redrawWindow(Render::CRenderContext& ctx, const PHLWINDOW& w, const SWinGeo& geo, const CRegion& clip, const PHLMONITOR& mon, const Vector2D& capturePos,
                          const Vector2D* canvas = nullptr) {
            const auto res = w->resource();
            if (!res)
                return;
            Vector2D base = geo.pos;
            if (!w->backend().isX11())
                base -= windowClientGeometryPos(w);
            static auto PBLUR = CConfigValue<Config::INTEGER>("decoration:blur:enabled");
            const bool  blur  = !canvas && !geo.opaque && *PBLUR && !(w->m_ruleApplicator && w->m_ruleApplicator->noBlur().valueOrDefault());
            const int   round = canvas ? int(std::lround(geo.round / zoomScale(mon))) : geo.round;
            const float alpha = windowAlpha(w);
            const float fade  = windowFade(w);
            res->breadthfirst(
                [&](SP<CWLSurfaceResource> surf, const Vector2D& local, void*) {
                    const auto tex = surf->m_current.texture;
                    if (!tex)
                        return;
                    const bool main = surf == res;
                    const CBox px   = CBox{base + local, surf->m_current.size}.translate(-mon->m_position).scale(mon->m_scale);
                    const auto box  = canvas ? px.copy().translate(-*canvas) : zoomed(mon, px).translate(-capturePos);
                    if (box.w < 1 || box.h < 1)
                        return;
                    drawSurfaceTexture(ctx, surf,
                                       CTexPassElement::SRenderData{
                                           .tex                   = tex,
                                           .box                   = box,
                                           .a                     = alpha,
                                           .blurA                 = fade, // the blur fades with the window, as in Hyprland
                                           .round                 = main ? round : 0,
                                           .roundingPower         = geo.roundingPower,
                                           .blur                  = main && blur,
                                           .blockBlurOptimization = true, // blur what is really under it (the cover), not the cached wallpaper
                                           // the damage clip alone isn't enough: the blur pass draws the blurred
                                           // background over the whole box unless clipRegion is set
                                           .clipRegion = clip,
                                       },
                                       clip);
                },
                nullptr);
        }

        // A window that wobbles isn't in its box: Hyprland draws it into a buffer of its own
        // and bends that through the window's transformers. Whatever goes over such a window
        // in the capture, its cover or the window itself drawn again, takes the same way, or
        // the window shows from behind a cover that stands still.
        // paint(canvas): draws it for the window standing still, into a buffer that starts
        // at the monitor pixel canvas.pos(). The result is drawn only within clip.
        // false: nothing was drawn (it doesn't bend after all, or there is no buffer).
        SP<Render::IFramebuffer> g_bentSource;

        template <class F>
        bool drawBent(Render::CRenderContext& ctx, const PHLWINDOW& w, const PHLMONITOR& mon, const Vector2D& capturePos, const CRegion& clip, const F& paint) {
            const auto& fx = w->effects();
            if (!fx.transformers())
                return false;
            // the box Hyprland bends: the window with its decorations, as renderWindow has it
            const auto  where   = w->presentation().renderPresentation();
            const CBox  current = w->getFullWindowBoundingBox().translate(where.workspaceOffset + where.floatingOffset - mon->m_position);
            const auto  plan    = fx.transformers()->plan(current, fx.transformedExtents(current));
            const CBox  canvas  = Render::pixelBoxForLogical(plan.sourceBox.copy().expand(1.0 / mon->m_scale), mon->m_scale);
            if (plan.stages.empty() || canvas.empty() || canvas.w > 16384 || canvas.h > 16384)
                return false;
            if (!g_bentSource)
                g_bentSource = g_pHyprRenderer->createFB("noshare-cover bent");
            if (!g_bentSource || !g_bentSource->alloc(int(canvas.w), int(canvas.h)))
                return false;
            if (const auto target = ctx.m_data.currentFB)
                g_bentSource->setImageDescription(target->imageDescription());

            Render::SWindowTransformBuffer bent;
            {
                const auto                          viewport = ctx.m_data.fbSize;
                const Hyprutils::Utils::CScopeGuard restoreViewport{[&] { g_pHyprRenderer->setViewport(0, 0, viewport.x, viewport.y); }};
                // what paint adds is in the buffer's pixels, not the capture's
                const Hyprutils::Utils::CScopeGuard restorePainted{[&, painted = g_painted] { g_painted = painted; }};
                const auto                          state = ctx.saveDrawState();
                const auto                          bound = g_pHyprRenderer->bindTempFB(ctx, g_bentSource);
                const CRegion                       all{CBox{{}, canvas.size()}};
                ctx.m_data.renderModif        = {};
                ctx.m_data.useNearestNeighbor = false;
                ctx.m_data.damage             = all;
                ctx.m_data.fbSize             = canvas.size();
                g_pHyprRenderer->setProjectionType(ctx, Render::RPT_EXPORT);
                g_pHyprRenderer->setViewport(0, 0, canvas.w, canvas.h);
                g_pHyprRenderer->draw(ctx, CClearPassElement::SClearData{{0.F, 0.F, 0.F, 0.F}}, all);
                paint(canvas);
                bent = fx.transformers()->transform(ctx, {.framebuffer = g_bentSource, .box = canvas}, plan,
                                                    Render::SWindowTransformContext{
                                                        .currentBox = current,
                                                        .inputBox   = plan.sourceBox,
                                                        .outputBox  = plan.outputBox,
                                                        .monitor    = mon,
                                                    });
            }
            const auto tex = bent.success && bent.framebuffer ? bent.framebuffer->getTexture() : nullptr;
            if (!tex)
                return false;
            g_painted.add(clip);
            g_pHyprRenderer->draw(ctx, CTexPassElement::SRenderData{.tex = tex, .box = zoomed(mon, bent.box).translate(-capturePos)}, clip);
            return true;
        }

    } // namespace

    // Everything drawn over Hyprland's image in this capture frame (capture pixels): covers,
    // effects, black boxes. A software cursor under it is drawn again (cursorOverCovers).
    CRegion g_painted;

    // One hidden surface's cover at box (capture pixels), only within clip (empty: the whole
    // box): its media, its own picture blurred or pixelated, or black when there is nothing
    // to show. what: the surface, for the trace.
    // Returns the core's id of the media it shows or waits for (0: none), for the pump.
    uint64_t paintCover(Render::CRenderContext& ctx, const SP<Render::ITexture>& image, const std::string& what, const SRuleValues& rules, const CBox& box, const Vector2D& capturePos,
                        int round, float roundingPower, const CRegion& clip) {
        const auto    kind  = coverKind(rules);
        const CRegion area  = clip.empty() ? CRegion{box} : clip;
        uint64_t      cover = 0;
        switch (kind.kind) {
            case SCoverKind::MEDIA: {
                const auto req = playRequest(rules, box.size());
                nsc_frame  f{};
                const bool ready = nsc_resolve(&req, &f);
                cover            = f.cover_id; // also while it is still being opened
                if (const auto tex = ready ? textureFor(f) : nullptr) {
                    NSC_TRACE("%s: cover %ux%u at %.0f,%.0f %.0fx%.0f\n", what.c_str(), f.width, f.height, box.x, box.y, box.w, box.h);
                    // black first: a picture with transparent pixels would show the window through them
                    drawBlack(ctx, box, round, roundingPower, area);
                    g_pHyprRenderer->draw(ctx, CTexPassElement::SRenderData{.tex = tex, .box = box, .round = round, .roundingPower = roundingPower}, area);
                    return cover;
                }
                NSC_TRACE("%s: no cover frame yet\n", what.c_str());
                break;
            }
            case SCoverKind::BLUR:
            case SCoverKind::PIXELATE:
                if (drawEffect(ctx, image, kind, box, capturePos, round, roundingPower, area)) {
                    NSC_TRACE("%s: %s %d at %.0f,%.0f %.0fx%.0f\n", what.c_str(), kind.kind == SCoverKind::BLUR ? "blur" : "pixelate", kind.amount, box.x, box.y, box.w, box.h);
                    g_painted.add(area);
                    return 0;
                }
                break;
            case SCoverKind::TRANSPARENT: break; // and still in the image
        }
        drawBlack(ctx, box, round, roundingPower, area);
        return cover;
    }

    // Where Hyprland draws a window among the others (renderWorkspaceWindows[Fullscreen]):
    // tiled, floating, and on a workspace with a fullscreen window that window over both, with
    // the floating windows that go over it (allowed there, or fading out under it) on top. The
    // special workspace the same way over the regular one (+4), pinned windows over everything.
    int windowStack(const PHLWINDOW& w) {
        if (windowPinned(w))
            return 8;
        const auto& fsc         = Fullscreen::controller();
        const bool  fsWorkspace = fsc && w->m_workspace && fsc->hasFullscreen(w->m_workspace);
        int         stack       = w->isFloating() ? 1 : 0;
        if (fsWorkspace && fsc->isFullscreen(w))
            stack = 2;
        else if (fsWorkspace && stack == 1 && w->shouldRenderOverFullscreen())
            stack = 3;
        return (w->onSpecialWorkspace() ? 4 : 0) + stack;
    }

    std::optional<SWinGeo> windowGeo(const PHLWINDOW& w, const PHLMONITOR& mon, const Vector2D& capturePos) {
        if (!g_pHyprRenderer->shouldRenderWindow(w, mon) || w->isHidden())
            return std::nullopt;
        const auto  fade = windowFade(w);
        const auto* ws   = w->m_workspace.get();
        // nothing on screen: no workspace, or faded out behind a fullscreen or maximized
        // window (shouldRenderWindow still lets those through while the workspace slides in)
        if (!ws || fade == 0.F)
            return std::nullopt;

        SWinGeo    g;
        g.size = w->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
        g.pos  = w->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT) + windowRenderOffset(w);
        g.box = zoomed(mon, CBox{g.pos.x, g.pos.y, std::max(g.size.x, 5.0), std::max(g.size.y, 5.0)}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
        if (g.box.w < 1 || g.box.h < 1)
            return std::nullopt;
        const auto& fsc        = Fullscreen::controller();
        const bool  fullscreen = fsc && fsc->isFullscreen(w, Fullscreen::FSMODE_FULLSCREEN);
        const bool  region     = capturePos != Vector2D{};
        g.corner               = fullscreen ? 0 : int(std::lround(w->presentation().rounding() * mon->m_scale * zoomScale(mon)));
        g.round                = region ? 0 : g.corner;
        g.roundingPower        = region || fullscreen ? 2.F : w->presentation().roundingPower();
        g.layer                = windowStack(w);
        // Hyprland's opaque() ignores the layout alpha: a group tab or monocle window that
        // is fading out on a switch would still count as opaque and cut the cover below it
        g.opaque = fade >= 1.F && w->presentation().opaque();
        g.area   = g.box;
        if (w->effects().hasActiveTransformers()) {
            const CBox still{g.pos - mon->m_position, {std::max(g.size.x, 5.0), std::max(g.size.y, 5.0)}};
            const CBox bent = w->effects().transformedExtents(still);
            if (bent != still) {
                g.bent = true;
                g.area = zoomed(mon, bent.copy().scale(mon->m_scale)).translate(-capturePos);
                // its box says nothing about what it hides: it is drawn again over the covers
                // below it, bent as it is (see paintCovers)
                g.opaque = false;
            }
        }
        return g;
    }

    // Hyprland's black boxes are switched off for the frame (see suppressNoScreenShare), so
    // everything without a cover is drawn black here.
    // revealed: hidden surfaces this client sees as they are (see revealFor).
    // left: surfaces with a transparent cover were left out of this frame's image (see
    // CSceneSwap), so nothing goes over them; otherwise they get the black box.
    // image: what Hyprland draws the frame from, for blur and pixelate.
    void paintCovers(Render::CRenderContext& ctx, Screenshare::CScreenshareFrame* frame, const SCaptureClient& client, const std::unordered_set<uintptr_t>& revealed, bool left,
                     const SP<Render::ITexture>& image) {
        if (!frame || !frame->m_session || !g_pHyprRenderer) {
            NSC_TRACE("skip: frame %p session %d renderer %d\n", static_cast<void*>(frame), frame && frame->m_session ? 1 : 0, g_pHyprRenderer ? 1 : 0);
            return;
        }
        const auto mon = ctx.m_data.pMonitor.lock();
        if (!mon) {
            NSC_TRACE("skip: no monitor in render data\n");
            return;
        }

        pushSettings();
        nsc_begin_frame();
        // the session is owned by the manager: its weak pointer is only read (see captureClient)
        g_fxStream = reinterpret_cast<uintptr_t>(frame->m_session.get());
        {
            auto& s = g_fx[mon->m_id].streams[g_fxStream];
            ++s.frames;
            s.seen = g_frameCount;
        }
        // Rebuild this session's covers on this monitor (see SAnimBox), drop those of sessions
        // that don't capture it any more, leave the rest alone: a screenshot from a show_to
        // client draws none of a stream's video covers, and the stream's video would freeze.
        std::unordered_set<uint64_t> asked; // the covers on it the pump already asks about
        for (const auto& a : g_animBoxes)
            if (a.mon.lock() == mon)
                asked.insert(a.cover);
        std::erase_if(g_animBoxes, [&](const SAnimBox& a) { return a.mon.expired() || (a.mon.lock() == mon && (a.owner == g_fxStream || !capturedBy(mon, a.owner))); });
        NSC_TRACE("frame: monitor %s\n", mon->m_name.c_str());

        const auto capturePos = frame->m_session->m_captureBox.pos();
        const auto now        = std::chrono::steady_clock::now();

        // What is covered here, for the closing covers (trackClosed) and the hold. Kept for
        // left out surfaces too: their fade-out on close and the hold.
        std::unordered_set<uintptr_t> seen;
        const auto                    remember = [&](uintptr_t key, SLastCover&& last) {
            seen.insert(key);
            if (!g_held.contains(key)) // a held cover keeps the time and rules it had
                g_lastCovers[key] = std::move(last);
        };

        // Windows drawn on top of others: later in Hyprland's stacking (see windowStack), then
        // in list order. The cover of a closing window goes among them where the window was
        // when it closed, then where Hyprland draws its fade-out once that is found: after
        // the windows of the same plane.
        // A cover is only drawn where its window is actually visible, so an opaque window
        // placed over a hidden one stays visible. Layers (bars) are left alone: they are
        // often translucent and we can't tell.
        struct SVisible {
            PHLWINDOW                  w; // none for a closing window's cover
            SWinGeo                    geo;
            std::optional<SRuleValues> rules;        // set for hidden windows and closing covers
            bool                       gone = false; // left out of the image, as if it wasn't there
        };
        std::vector<SVisible> visible;
        for (const auto& w : Desktop::windowState()->windows()) {
            const auto g = w ? windowGeo(w, mon, capturePos) : std::nullopt;
            if (!g)
                continue;
            auto& v = visible.emplace_back(SVisible{.w = w, .geo = *g});
            if (w->m_ruleApplicator && w->m_ruleApplicator->noScreenShare().valueOrDefault()) {
                v.rules = ruleValuesFor(w);
                v.gone  = left && transparentCover(*v.rules);
                remember(reinterpret_cast<uintptr_t>(w.get()),
                         SLastCover{
                             .win           = w,
                             .mon           = mon,
                             .box           = CBox{g->pos, g->size},
                             .rules         = *v.rules,
                             .rounding      = w->presentation().rounding(),
                             .roundingPower = w->presentation().roundingPower(),
                             .stack         = g->layer,
                             .at            = now,
                         });
            }
        }

        // Layers (layer-shell) with no_screen_share: same geometry as Hyprland's black rect,
        // no rounding. Drawn in the layer pass below.
        struct SLayerCover {
            PHLLS       l;
            SRuleValues rules;
            CBox        box;           // capture pixels
            bool        done = false; // drawn (or left out) in the layer pass
        };
        std::unordered_map<uintptr_t, SLayerCover> layerCovers;
        for (const auto& l : Desktop::layerState()->layers()) {
            if (!l || !l->m_ruleApplicator || !l->m_ruleApplicator->noScreenShare().valueOrDefault() || !viewVisible(l) || l->m_monitor.lock() != mon)
                continue;
            const auto pos  = l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
            const auto size = l->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
            const auto box  = zoomed(mon, CBox{pos.x, pos.y, std::max(size.x, 5.0), std::max(size.y, 5.0)}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
            if (box.w < 1 || box.h < 1)
                continue;
            const auto key   = reinterpret_cast<uintptr_t>(l.get());
            auto       rules = ruleValuesFor(l);
            remember(key,
                     SLastCover{
                         .layer   = l,
                         .isLayer = true,
                         .mon     = mon,
                         .box     = CBox{pos, size},
                         .rules   = rules,
                         .stack   = int(l->m_layer),
                         .at      = now,
                     });
            layerCovers.emplace(key, SLayerCover{.l = l, .rules = std::move(rules), .box = box});
        }

        // Surfaces shown as they are to this client (or held for the others only) are still
        // covered for the other clients: keep their entries, and the time they were last
        // covered, for the closing covers in the others' frames.
        seen.insert(revealed.begin(), revealed.end());
        trackClosed(mon, seen, now);
        const auto closing = closingCovers(mon, capturePos, now, client, left);
        for (const auto& c : closing) {
            if (c.isLayer)
                continue;
            SWinGeo g;
            g.box           = c.box;
            g.area          = c.box;
            g.round         = c.round;
            g.roundingPower = c.roundingPower;
            g.layer         = c.stack;
            visible.push_back(SVisible{.geo = g, .rules = c.rules});
        }

        // o is stacked above me
        const auto isAbove = [&](size_t o, size_t me) {
            const auto& a = visible[o].geo;
            const auto& b = visible[me].geo;
            return o != me && (a.layer > b.layer || (a.layer == b.layer && o > me));
        };
        const auto visibleRegion = [&](size_t i) {
            CRegion region{visible[i].geo.area};
            // Only opaque windows hide what is below them. Under a translucent one the
            // hidden window would show through, so the cover goes there too and the
            // translucent window is drawn again on top of it (see below).
            for (size_t j = 0; j < visible.size(); ++j)
                if (isAbove(j, i) && visible[j].geo.opaque && !visible[j].gone)
                    subtractRounded(region, visible[j].geo.box, visible[j].geo.corner);
            return region;
        };
        std::vector<std::pair<size_t, CRegion>> drawn; // covers (or black) drawn, per window

        // bottom to top, as Hyprland stacks them: where two covers overlap, the upper one stays
        std::vector<size_t> order(visible.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::ranges::stable_sort(order, {}, [&](size_t k) { return visible[k].geo.layer; });

        for (const size_t i : order) {
            if (!visible[i].rules || visible[i].gone)
                continue;
            const auto& w     = visible[i].w;
            const auto& rules = *visible[i].rules;
            const auto& geo   = visible[i].geo;
            if (!w) { // a closing window's cover
                const CRegion clip = visibleRegion(i);
                if (clip.empty())
                    continue;
                paintCover(ctx, image, "closing window", rules, geo.box, capturePos, geo.round, geo.roundingPower, clip);
                drawn.emplace_back(i, clip);
                continue;
            }
            const auto pos           = geo.pos;
            const auto size          = geo.size;
            const auto box           = geo.box;
            const int  round         = geo.round;
            const auto roundingPower = geo.roundingPower;

            const CRegion clip = visibleRegion(i);
            if (clip.empty()) {
                NSC_TRACE("window %s: fully under other windows\n", windowClass(w).c_str());
            }
            if (!w->backend().isX11())
                paintPopupBoxes(ctx, w, pos - windowClientGeometryPos(w), mon, capturePos, image, coverKind(rules));
            if (clip.empty())
                continue;
            const auto what  = "window " + windowClass(w);
            uint64_t   cover = 0;
            // A cover bends with its window: it is painted in a buffer where the window stands
            // and takes the window's way from there. Blur and pixelate are made of the image,
            // which has the window bent already: they go straight over where it is.
            const auto kind   = coverKind(rules).kind;
            const bool effect = kind == SCoverKind::BLUR || kind == SCoverKind::PIXELATE;
            const bool bent   = geo.bent && !effect && drawBent(ctx, w, mon, capturePos, clip, [&](const CBox& canvas) {
                const CBox at = CBox{pos.x, pos.y, std::max(size.x, 5.0), std::max(size.y, 5.0)}.translate(-mon->m_position).scale(mon->m_scale).translate(-canvas.pos());
                cover         = paintCover(ctx, image, what, rules, at, {}, int(std::lround(round / zoomScale(mon))), roundingPower);
            });
            if (geo.bent && effect)
                cover = paintCover(ctx, image, what, rules, geo.area, capturePos, 0, 2.F, clip);
            else if (!bent)
                cover = paintCover(ctx, image, what, rules, box, capturePos, round, roundingPower, clip);
            // media has the monitor repainted for its new frames (see pumpTick)
            if (cover)
                g_animBoxes.push_back({mon, CBox{pos.x, pos.y, size.x, size.y}, cover, g_fxStream});
            drawn.emplace_back(i, clip);
        }

        // Windows above a hidden one, bottom to top: draw them again over the cover, only
        // over covers below them, minus whatever is above them. A translucent window all
        // over them, so what shows through it (blurred, if it blurs) is the cover and never
        // the hidden window. An opaque one only in its rounded corners: a cover below it is
        // cut along its box, corners left in (subtractRounded), and the rounding is drawn
        // again here over the cover there (square in a region capture, see SWinGeo::corner:
        // the corners then show the window, never what was below them).
        if (!drawn.empty()) {
            for (const size_t j : order) {
                // Never redraw a hidden window from its real content. This also catches
                // windows that are only translucent for a moment (fading in on open).
                const auto& wj = visible[j].w;
                if (!wj || !wj->m_ruleApplicator || wj->m_ruleApplicator->noScreenShare().valueOrDefault())
                    continue;
                CRegion region;
                for (const auto& [i, clip] : drawn)
                    if (isAbove(j, i))
                        region.add(clip);
                if (region.empty())
                    continue;
                region.intersect(CRegion{visible[j].geo.area});
                for (size_t k = 0; k < visible.size(); ++k)
                    if (isAbove(k, j) && visible[k].geo.opaque && !visible[k].gone)
                        subtractRounded(region, visible[k].geo.box, visible[k].geo.corner);
                for (const auto& [i, clip] : drawn)
                    if (isAbove(i, j))
                        region.subtract(clip);
                if (region.empty())
                    continue;
                NSC_TRACE("window %s: redrawn over a cover\n", windowClass(visible[j].w).c_str());
                const bool bent = visible[j].geo.bent && drawBent(ctx, wj, mon, capturePos, region, [&](const CBox& canvas) {
                    const auto origin = canvas.pos();
                    redrawWindow(ctx, wj, visible[j].geo, CRegion{CBox{{}, canvas.size()}}, mon, capturePos, &origin);
                });
                if (!bent)
                    redrawWindow(ctx, wj, visible[j].geo, region, mon, capturePos);
            }
        }

        // Layers in Hyprland's order: background and bottom, top and overlay, each in its own
        // order, the fade-outs of a level after its layers. A hidden background or bottom
        // layer is covered over its whole area, the windows on it included, the same as
        // Hyprland's own black box. A top or overlay layer goes over everything below it:
        // hidden, its cover does; otherwise it is drawn again over the covers below it, and
        // a cover above it goes over it in turn. The popups of hidden layers get their boxes
        // over all of it, as Hyprland draws them last.
        CRegion coverArea; // every cover so far
        for (const auto& [i, clip] : drawn)
            coverArea.add(clip);
        std::vector<PHLLS> popupsOf;
        const auto         coverLayer = [&](SLayerCover& c) {
            c.done = true;
            if (left && transparentCover(c.rules))
                return;
            popupsOf.push_back(c.l);
            const auto pos = c.l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
            if (const auto cover = paintCover(ctx, image, "layer " + c.l->m_namespace, c.rules, c.box, capturePos, 0, 2.F))
                g_animBoxes.push_back({mon, CBox{pos, c.l->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT)}, cover, g_fxStream});
            coverArea.add(c.box);
        };
        for (uint32_t level = 0; level < mon->m_layerSurfaceLayers.size(); ++level) {
            for (const auto& ref : mon->m_layerSurfaceLayers[level]) {
                const auto l = ref.lock();
                if (!l || !l->m_ruleApplicator || !viewVisible(l) || l->m_monitor.lock() != mon)
                    continue;
                const auto pos = l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
                if (const auto it = layerCovers.find(reinterpret_cast<uintptr_t>(l.get())); it != layerCovers.end()) {
                    coverLayer(it->second);
                    continue;
                }
                if (level < 2 || l->m_ruleApplicator->noScreenShare().valueOrDefault() || coverArea.empty())
                    continue;
                // what is above windows (bars, launchers, notifications) and was covered
                CRegion region{zoomed(mon, CBox{pos, l->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT)}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos)};
                region.intersect(coverArea);
                if (region.empty())
                    continue;
                NSC_TRACE("layer %s: redrawn over a cover\n", l->m_namespace.c_str());
                redrawLayer(ctx, l, region, mon, capturePos);
            }
            for (const auto& c : closing) {
                if (!c.isLayer || c.stack != int(level))
                    continue;
                paintCover(ctx, image, "closing layer", c.rules, c.box, capturePos, c.round, c.roundingPower);
                coverArea.add(c.box);
            }
        }
        // not in the monitor's lists (Hyprland doesn't draw it then): on top, just in case
        for (auto& c : layerCovers | std::views::values)
            if (!c.done)
                coverLayer(c);
        for (const auto& l : popupsOf)
            paintPopupBoxes(ctx, l, l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT) - l->m_geometry.pos(), mon, capturePos, image,
                            coverKind(layerCovers.at(reinterpret_cast<uintptr_t>(l.get())).rules));

        paintExtraRects(ctx, mon, capturePos, revealed);
        nsc_end_frame();
        // a cover the pump hasn't asked about yet: soon, it may be a video still loading
        const bool fresh = std::ranges::any_of(g_animBoxes, [&](const SAnimBox& a) { return !asked.contains(a.cover) && a.mon.lock() == mon; });
        // a hold to end (see holdUnhidden), which may have started with nothing captured
        const bool holding = std::ranges::any_of(g_hidden, [](const auto& kv) { return kv.second.since.has_value(); });
        if (!g_animBoxes.empty() || !g_closing.empty() || holding)
            startPump(fresh);
        pruneTextures();
        drainNotifications();
    }

    void releasePaint() {
        g_bentSource.reset();
        g_fx.clear();
        g_textures.clear();
    }

} // namespace nsc
