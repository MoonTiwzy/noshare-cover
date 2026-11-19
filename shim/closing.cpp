// Covers that outlive what they cover: closing windows and layers, surfaces that just
// stopped being hidden; and the pump that keeps animated and closing covers repainted.

#include "shim.hpp"

namespace nsc {

    namespace {

        SP<CEventLoopTimer> g_pump;
        constexpr auto      PUMP_EVERY       = std::chrono::milliseconds(16);
        // The longest the pump sleeps while a cover is on a captured monitor: asking the core
        // about a video is what keeps it decoding, and it dozes off after 400 ms unasked.
        constexpr int PUMP_LONGEST_MS = 250;
        constexpr auto      FADE_BIND_WINDOW = std::chrono::milliseconds(150);

        // Something last covered longer ago than this gets a closing cover only if its
        // fade-out is found (see trackClosed).
        constexpr auto STALE_COVER = std::chrono::seconds(1);

        bool isWindowFade(const SP<Desktop::IFadeout>& f) {
            const auto plane = f->plane();
            return plane == Desktop::FADEOUT_PLANE_WINDOW_TILED || plane == Desktop::FADEOUT_PLANE_WINDOW_FLOATING || plane == Desktop::FADEOUT_PLANE_WINDOW_OVER_FULLSCREEN;
        }

        bool isLayerFade(const SP<Desktop::IFadeout>& f) {
            const auto plane = f->plane();
            return plane == Desktop::FADEOUT_PLANE_LAYER_BACKGROUND || plane == Desktop::FADEOUT_PLANE_LAYER_BOTTOM || plane == Desktop::FADEOUT_PLANE_LAYER_TOP ||
                plane == Desktop::FADEOUT_PLANE_LAYER_OVERLAY;
        }

        // The fade-out Hyprland created for a closed window or layer: one of the same kind
        // on the same monitor, not taken by another closing cover, nearest to where it was.
        // The match is fuzzy: IFadeout doesn't keep its source, so this goes by plane and
        // geometry and can pick the fade-out of something else that closed nearby (e.g. for
        // a layer that went invisible without layer.closed), and then the cover follows the
        // wrong fade-out. Hyprland marks the fade-out of a hidden surface as such, and those
        // come first: the nearest fade-out of a window that isn't hidden is taken only when
        // there is no such one. A closing window leaves fullscreen and its group while still
        // mapped, after window.close (where its cover starts) and before its fade-out is
        // made, and its rules are matched again then: one that depends on fullscreen or the
        // group can stop matching, and the fade-out is not marked. (Focus can't do it: the
        // window is unmapped by the time focus moves on, and an unmapped window's rules
        // aren't matched again.) A surface closed during its hold (see holdUnhidden) wasn't
        // hidden any more by then: for it the other way round.
        SP<Desktop::IFadeout> findFadeFor(const SClosing& c, const PHLMONITOR& mon) {
            const auto& state = Desktop::fadingOutState();
            if (!state)
                return nullptr;
            SP<Desktop::IFadeout> best, other;
            const double          range    = std::max(c.box.w, c.box.h);
            double                bestDist = range, otherDist = range;
            const auto            center   = c.box.middle();
            for (const auto& f : state->fadeouts()) {
                if (!f || f->done() || !(c.isLayer ? isLayerFade(f) : isWindowFade(f)) || f->monitor().lock() != mon)
                    continue;
                if (std::ranges::any_of(g_closing, [&](const SClosing& o) { return o.fade.lock() == f; }))
                    continue;
                const double d = f->geometricBox(Desktop::View::IGeometric::GEOMETRIC_CURRENT).middle().distance(center);
                if (f->source().noScreenShare != c.held) {
                    if (d <= bestDist) {
                        best     = f;
                        bestDist = d;
                    }
                } else if (d <= otherDist) {
                    other     = f;
                    otherDist = d;
                }
            }
            return best ? best : other;
        }

        // Where Hyprland draws a window's fade-out among the windows, as SWinGeo::layer counts
        // them: after the live windows of its plane (renderWorkspaceWindows[Fullscreen]), the
        // special workspace's over the regular one.
        int fadeStack(const SP<Desktop::IFadeout>& f) {
            const auto ws   = f->workspace().lock();
            const int  base = ws && ws->type() == Workspace::eWorkspaceType::SPECIAL ? 4 : 0;
            switch (f->plane()) {
                case Desktop::FADEOUT_PLANE_WINDOW_TILED: return base;
                case Desktop::FADEOUT_PLANE_WINDOW_OVER_FULLSCREEN: return base + 3;
                default: return base + 1; // floating
            }
        }

        // A fade-out found for c, and where c's cover goes with it (until then: where the
        // window was when it closed).
        void bindFade(SClosing& c, const SP<Desktop::IFadeout>& f) {
            c.fade  = f;
            c.bound = !!f;
            if (f && !c.isLayer)
                c.stack = fadeStack(f);
        }

        // Hyprland emits window.close / layer.closed from unmap, while the window or layer
        // still has its geometry and rules and before its fade-out exists. Starting the
        // closing cover here doesn't depend on the stream sending frames (a static screen
        // sends none, and the frame-based tracking above then only has old data).
        // stack: see SClosing
        void startClosing(bool isLayer, uintptr_t key, const PHLMONITOR& mon, const CBox& box, SRuleValues&& rules, float rounding, float roundingPower, bool held, int stack) {
            if (!beingCaptured(mon))
                return;
            g_lastCovers.erase(key); // don't start it a second time from the next frame
            g_closing.push_back(SClosing{
                .isLayer       = isLayer,
                .mon           = mon,
                .box           = box,
                .rules         = std::move(rules),
                .rounding      = rounding,
                .roundingPower = roundingPower,
                .held          = held,
                .stack         = stack,
                .bindUntil     = std::chrono::steady_clock::now() + FADE_BIND_WINDOW,
            });
            NSC_TRACE("%s close event: cover kept at %.0f,%.0f %.0fx%.0f\n", isLayer ? "layer" : "window", box.x, box.y, box.w, box.h);
            startPump();
        }

        // Covers are opened on a loader thread, so the first capture of a new cover
        // would only get the black box: a screenshot has no later frame to catch up
        // in. Start opening as soon as a hidden window or layer gets its rules.
        void prewarm(const SRuleValues& rules) {
            if (coverKind(rules).kind != SCoverKind::MEDIA)
                return;
            // settings normally arrive with the first capture frame; without them the
            // warmed cover would be dropped as soon as they do (settings change = reset)
            pushSettings();
            const auto req = playRequest(rules);
            nsc_prewarm(&req);
        }

        // A surface's entry in g_hidden, if it is this surface's (a new one may get the address
        // of one that is gone).
        template <class V>
        SHidden* hiddenEntry(const V& v) {
            const auto it = g_hidden.find(reinterpret_cast<uintptr_t>(v.get()));
            if (it == g_hidden.end())
                return nullptr;
            if constexpr (std::is_same_v<V, PHLWINDOW>)
                return it->second.win.lock() == v ? &it->second : nullptr;
            else
                return it->second.layer.lock() == v ? &it->second : nullptr;
        }

        // The surface's entry, still within its hold (see holdUnhidden), by address.
        const SHidden* inHold(uintptr_t key, std::chrono::steady_clock::time_point now) {
            const auto it = g_hidden.find(key);
            return it != g_hidden.end() && it->second.since && now - *it->second.since < holdFor(it->second.rules) ? &it->second : nullptr;
        }

        void forgetGone() {
            std::erase_if(g_hidden, [](const auto& kv) { return kv.second.win.expired() && kv.second.layer.expired(); });
        }

        template <class V>
        SHidden* track(const V& v, bool hidden, std::chrono::steady_clock::time_point now) {
            const auto key   = reinterpret_cast<uintptr_t>(v.get());
            auto*      entry = hiddenEntry(v);
            if (!entry)
                g_hidden.erase(key); // whatever had this address before
            if (hidden) {
                SHidden h{.rules = ruleValuesFor(v)};
                if constexpr (std::is_same_v<V, PHLWINDOW>)
                    h.win = v;
                else
                    h.layer = v;
                return &(g_hidden[key] = std::move(h));
            }
            if (!entry || entry->since)
                return entry;
            entry->since = now;
            NSC_TRACE("%s: not hidden any more, its cover held for %lld ms\n", std::is_same_v<V, PHLWINDOW> ? "window" : "layer",
                      static_cast<long long>(holdFor(entry->rules).count()));
            startPump(); // to repaint it once the hold is over
            return entry;
        }

        void pumpTick(SP<CEventLoopTimer> self, void*) {
            // ms until there is something to repaint again: 0 = there was now, -1 = never
            int        wait   = -1;
            const auto sooner = [&](int ms) { wait = wait < 0 ? ms : std::min(wait, ms); };
            const auto shared = [](const PHLMONITORREF& ref) { return beingCaptured(ref.lock()); };
            if (g_pHyprRenderer) {
                // A cover's area is repainted when the cover has a new frame, so a 24 fps video
                // has the monitor render 24 times a second, and a still not at all (only when its
                // file was replaced). The core is asked about every cover on a captured monitor,
                // which also keeps a video decoding from one frame to the next and has the core
                // look at the file of a still one. Only while the capture it was drawn for
                // still runs (see SAnimBox): another one may see the surface as it is.
                for (const auto& a : g_animBoxes) {
                    if (!capturedBy(a.mon.lock(), a.owner))
                        continue;
                    const int ms = nsc_next_frame_ms(a.cover);
                    if (ms < 0)
                        continue;
                    if (ms == 0)
                        g_pHyprRenderer->damageBox(a.box);
                    sooner(ms);
                }
                // Closing covers: keep repainting while they're up, and once more after
                // they expire so the stream drops them.
                const auto now = std::chrono::steady_clock::now();
                for (const auto& c : g_closing) {
                    if (!shared(c.mon))
                        continue;
                    g_pHyprRenderer->damageBox(c.box);
                    sooner(0);
                }
                std::erase_if(g_closing, [&](const SClosing& c) { return !shared(c.mon) || (c.holdUntil && now >= *c.holdUntil); });
                // Surfaces that stopped being hidden: repainted once their hold is over, or a
                // still monitor would keep their cover in the stream until something else
                // repaints it; twice, like a closing cover, for a client that asks for its
                // next frame only after the first one. A surface hidden again has no end of a
                // hold any more.
                for (auto it = g_hidden.begin(); it != g_hidden.end();) {
                    auto&      h = it->second;
                    const auto w = h.win.lock();
                    const auto l = h.layer.lock();
                    if (!w && !l) {
                        it = g_hidden.erase(it);
                        continue;
                    }
                    if (!h.since) {
                        ++it;
                        continue;
                    }
                    const auto mon      = w ? w->m_monitor.lock() : l->m_monitor.lock();
                    const bool captured = beingCaptured(mon);
                    const auto end      = *h.since + holdFor(h.rules);
                    if (now < end) {
                        if (captured)
                            sooner(int(std::chrono::ceil<std::chrono::milliseconds>(end - now).count()));
                        ++it;
                        continue;
                    }
                    if (captured) {
                        if (!h.repainted)
                            NSC_TRACE("pump: hold over, %s repainted\n", w ? "window" : "layer");
                        g_pHyprRenderer->damageMonitor(mon);
                        if (w)
                            g_pHyprRenderer->damageWindow(w, true); // where it reaches past its monitor
                        sooner(0);
                    }
                    if (captured && !std::exchange(h.repainted, true)) {
                        ++it;
                        continue;
                    }
                    it = g_hidden.erase(it);
                }
            }
            if (wait < 0) {
                NSC_TRACE("pump: idle, nothing to repaint or nothing captured\n");
                g_animBoxes.clear();
                g_closing.clear();
                self->updateTimeout(std::nullopt); // sharing stopped, go idle
                return;
            }
            // A frame stays ready until a capture takes it, and a client that asks for its next
            // frame only later needs the damage once more: look again at the old fixed pace.
            // With nothing ready, sleep until the next frame is.
            self->updateTimeout(wait == 0 ? PUMP_EVERY : std::chrono::milliseconds(std::min(wait, PUMP_LONGEST_MS)));
        }

        void stopPump() {
            if (!g_pump)
                return;
            g_pump->cancel();
            if (g_pEventLoopManager)
                g_pEventLoopManager->removeTimer(g_pump);
            g_pump.reset();
        }

    } // namespace

    std::vector<SAnimBox>                     g_animBoxes;
    std::unordered_map<uintptr_t, SLastCover> g_lastCovers;
    std::vector<SClosing>                     g_closing;
    // Surfaces kept covered for a moment after they stopped being hidden (see holdUnhidden),
    // with the rules they had while hidden. Valid only during one capture frame.
    std::unordered_set<uintptr_t>              g_held;
    std::unordered_map<uintptr_t, SRuleValues> g_heldRules;
    std::unordered_map<uintptr_t, SHidden>     g_hidden;

    // Covered windows and layers of this monitor that disappeared since the last frame:
    // closed ones get a closing cover, moved or hidden ones are just forgotten.
    void trackClosed(const PHLMONITOR& mon, const std::unordered_set<uintptr_t>& seen, std::chrono::steady_clock::time_point now) {
        for (auto it = g_lastCovers.begin(); it != g_lastCovers.end();) {
            const auto& last = it->second;
            const auto  m    = last.mon.lock();
            if (m && m != mon) {
                ++it;
                continue;
            }
            if (m && seen.contains(it->first)) {
                ++it;
                continue;
            }
            bool gone = false;
            if (last.isLayer) {
                const auto l = last.layer.lock();
                gone         = !l || !viewVisible(l);
            } else {
                const auto w = last.win.lock();
                gone         = !w || !w->mapped();
            }
            if (m && gone) {
                SClosing c{
                    .isLayer       = last.isLayer,
                    .mon           = mon,
                    .box           = last.box,
                    .rules         = last.rules,
                    .rounding      = last.rounding,
                    .roundingPower = last.roundingPower,
                    .held          = inHold(it->first, now) != nullptr,
                    .stack         = last.stack,
                    .bindUntil     = now + FADE_BIND_WINDOW,
                };
                // The stream may have had no frames for a while (static screen), so an old
                // entry can still be something closed just now. Keep it only if its fade-out
                // is running; otherwise it closed long ago and there's nothing to hide.
                const bool fresh = now - last.at < STALE_COVER;
                if (!fresh)
                    bindFade(c, findFadeFor(c, mon));
                if (fresh || c.bound) {
                    g_closing.push_back(std::move(c));
                    NSC_TRACE("%s closed: cover kept at %.0f,%.0f %.0fx%.0f\n", last.isLayer ? "layer" : "window", last.box.x, last.box.y, last.box.w, last.box.h);
                    startPump();
                }
            }
            it = g_lastCovers.erase(it);
        }
    }

    // A window that closes during its hold (see holdUnhidden) gets a closing cover too, with
    // the rules it had while hidden: its fade-out is a picture of what the hold covers.
    void onWindowClose(const PHLWINDOW& w) {
        if (!w || !w->m_ruleApplicator)
            return;
        const bool hidden = w->m_ruleApplicator->noScreenShare().valueOrDefault();
        const auto key    = reinterpret_cast<uintptr_t>(w.get());
        const auto held   = hidden ? nullptr : inHold(key, std::chrono::steady_clock::now());
        if (!hidden && (!held || held->win.lock() != w))
            return;
        const auto pos  = w->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT) + windowRenderOffset(w);
        const auto size = w->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
        // Where it is stacked now, for as long as its fade-out isn't found: with animations off,
        // or the noanim rule, Hyprland makes none (or one that is over at once), and over every
        // window the cover would hide the windows above it for the whole hold.
        startClosing(false, key, w->m_monitor.lock(), CBox{pos, size}, hidden ? ruleValuesFor(w) : SRuleValues{held->rules}, w->presentation().rounding(),
                     w->presentation().roundingPower(), !hidden, windowStack(w));
    }

    void prewarmWindow(const PHLWINDOW& w) {
        if (w && w->m_ruleApplicator && w->m_ruleApplicator->noScreenShare().valueOrDefault())
            prewarm(ruleValuesFor(w));
    }

    void prewarmLayer(const PHLLS& l) {
        if (l && l->m_ruleApplicator && l->m_ruleApplicator->noScreenShare().valueOrDefault())
            prewarm(ruleValuesFor(l));
    }

    void onLayerClose(const PHLLS& l) {
        if (!l || !l->m_ruleApplicator)
            return;
        const bool hidden = l->m_ruleApplicator->noScreenShare().valueOrDefault();
        const auto key    = reinterpret_cast<uintptr_t>(l.get());
        const auto held   = hidden ? nullptr : inHold(key, std::chrono::steady_clock::now());
        if (!hidden && (!held || held->layer.lock() != l))
            return;
        const auto pos  = l->position(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
        const auto size = l->size(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
        startClosing(true, key, l->m_monitor.lock(), CBox{pos, size}, hidden ? ruleValuesFor(l) : SRuleValues{held->rules}, 0.F, 2.F, !hidden, int(l->m_layer));
    }

    SHidden* trackHidden(const PHLWINDOW& w, bool hidden, std::chrono::steady_clock::time_point now) {
        return w ? track(w, hidden, now) : nullptr;
    }

    SHidden* trackHidden(const PHLLS& l, bool hidden, std::chrono::steady_clock::time_point now) {
        return l ? track(l, hidden, now) : nullptr;
    }

    void trackHidden(const PHLWINDOW& w) {
        forgetGone();
        if (w && w->m_ruleApplicator && w->mapped())
            track(w, w->m_ruleApplicator->noScreenShare().valueOrDefault(), std::chrono::steady_clock::now());
    }

    void trackHidden(const PHLLS& l) {
        forgetGone();
        if (l && l->m_ruleApplicator)
            track(l, l->m_ruleApplicator->noScreenShare().valueOrDefault(), std::chrono::steady_clock::now());
    }

    // The closing covers of mon to draw in this frame (paintCovers puts each where the surface
    // was stacked, then where Hyprland stacks its fade-out, see SClosing::stack). Clients that
    // may see the surface (show_to / hide_from) get none:
    // they see its fade-out as it is.
    // left: transparent surfaces were left out of this frame, their fade-outs too.
    std::vector<SClosingCover> closingCovers(const PHLMONITOR& mon, const Vector2D& capturePos, std::chrono::steady_clock::time_point now, const SCaptureClient& client,
                                             bool left) {
        std::vector<SClosingCover> out;
        for (auto& c : g_closing) {
            if (c.mon.lock() != mon)
                continue;
            if (!c.holdUntil) {
                auto fade = c.fade.lock();
                // once found, never looked for again: a fade-out that is gone has ended, and
                // another one nearby is someone else's
                if (!fade && !c.bound && now < c.bindUntil) {
                    fade = findFadeFor(c, mon);
                    bindFade(c, fade);
                }
                if (fade && !fade->done())
                    c.box = fade->geometricBox(Desktop::View::IGeometric::GEOMETRIC_CURRENT);
                else if (c.bound || now >= c.bindUntil)
                    // Without a fade-out the hold starts when the search for one ends, right
                    // after the close, not at the next frame: with single shots (nothing
                    // renders in between) that can be seconds later. A fade-out that ended
                    // counts from now, also when Hyprland has dropped it already (it does at
                    // its next tick, often before the next capture frame): a late extra cover
                    // when no frame came for a while, never a missing one. From the end of
                    // the search instead, a close animation longer than it lost the hold.
                    c.holdUntil = (c.bound ? now : c.bindUntil) + holdFor(c.rules);
            }
            if (c.holdUntil && now >= *c.holdUntil)
                continue; // the pump drops it and repaints
            if (client.sees(c.rules))
                continue;
            if (left && transparentCover(c.rules))
                continue;

            const auto box =
                zoomed(mon, CBox{c.box.x, c.box.y, std::max(c.box.w, 5.0), std::max(c.box.h, 5.0)}.translate(-mon->m_position).scale(mon->m_scale)).translate(-capturePos);
            if (box.w < 1 || box.h < 1)
                continue;
            const int round = capturePos != Vector2D{} ? 0 : int(std::lround(c.rounding * mon->m_scale * zoomScale(mon)));
            out.push_back(SClosingCover{
                .rules         = c.rules,
                .box           = box,
                .round         = round,
                .roundingPower = c.roundingPower,
                .isLayer       = c.isLayer,
                .stack         = c.stack,
            });
        }
        return out;
    }

    void startPump(bool soon) {
        if (!g_pEventLoopManager)
            return;
        if (!g_pump) {
            g_pump = makeShared<CEventLoopTimer>(PUMP_EVERY, pumpTick, nullptr);
            g_pEventLoopManager->addTimer(g_pump);
        } else if (!g_pump->armed() || (soon && g_pump->leftUs() > float(std::chrono::duration_cast<std::chrono::microseconds>(PUMP_EVERY).count())))
            g_pump->updateTimeout(PUMP_EVERY);
    }

    void releaseClosing() {
        stopPump();
        g_lastCovers.clear();
        g_closing.clear();
        g_hidden.clear();
    }

} // namespace nsc
