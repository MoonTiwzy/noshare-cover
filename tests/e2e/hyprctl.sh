#!/usr/bin/env bash
# `hyprctl clients` and `hyprctl activewindow` show what the plugin does to each window
# (noScreenShare, noScreenShareCover), in both formats, and are Hyprland's own again once
# the plugin is unloaded. With another plugin that wraps `clients` too, before or after
# this one, unloading either first neither crashes nor leaves a wrapper of an unloaded
# plugin to be called or destroyed.
#
#   tests/e2e/hyprctl.sh <libnoshare-cover.so>
#
# Requires: Hyprland, foot, jq, imagemagick, c++ and pkg-config with Hyprland's headers (the
# other plugin is built at runtime); a seat. Exit code 0 means all passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-ctl.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }
# $1: what, $2: got, $3: expected
same() { [ "$2" = "$3" ] && pass "$1: $2" || fail "$1: got '$2', expected '$3'"; }

magick -size 64x64 xc:'#ff00ff' "$WORK/cover.png"
magick -size 64x64 xc:'#ff8800' "$WORK/rule \"1\".png"

cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/cover.png" } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true },
    debug = { suppress_errors = true },
})
hl.window_rule({ match = { class = "ctl-default" }, no_screen_share = true })
hl.window_rule({ match = { class = "ctl-rule" }, no_screen_share = true, no_screen_share_cover = [[$WORK/rule "1".png]] })
hl.window_rule({ match = { class = "ctl-blur" }, no_screen_share = true, no_screen_share_cover = "blur" })
hl.window_rule({ match = { class = "ctl-pixelate" }, no_screen_share = true, no_screen_share_cover = "pixelate 8" })
hl.window_rule({ match = { class = "ctl-gone" }, no_screen_share = true, no_screen_share_cover = "transparent" })
LUA

hctl() { hyprctl -i 0 "$@"; }
# field of a window by class: JSON, then the plain format
json() { hctl clients -j | jq -r --arg c "$1" --arg f "$2" '.[] | select(.class == $c) | .[$f]'; }
plain() { hctl clients | awk -v c="$1" -v f="$2" '
    /^Window / { hit = 0 } $0 == "\tclass: " c { hit = 1 }
    hit && index($0, "\t" f ": ") == 1 { print substr($0, length(f) + 4) }'; }

pkill -xu "$(id -u)" Hyprland; sleep 0.5
Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
export WAYLAND_DISPLAY=$(ls "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
sleep 2

CLASSES="ctl-default ctl-rule ctl-blur ctl-pixelate ctl-gone ctl-plain"
for c in $CLASSES; do
    foot --app-id "$c" sh -c 'while :; do sleep 1; done' >/dev/null 2>&1 &
done
for _ in $(seq 60); do sleep 0.25; [ "$(hctl clients -j | jq length)" = 6 ] && break; done
sleep 0.5

echo "== hyprctl clients"
hctl clients -j | jq -e . >/dev/null && pass "the JSON parses" || fail "the JSON doesn't parse"
check() { # $1: class, $2: hidden, $3: cover
    same "$1 noScreenShare (json)" "$(json "$1" noScreenShare)" "$2"
    same "$1 noScreenShareCover (json)" "$(json "$1" noScreenShareCover)" "$3"
    same "$1 noScreenShare" "$(plain "$1" noScreenShare)" "$([ "$2" = true ] && echo 1 || echo 0)"
    same "$1 noScreenShareCover" "$(plain "$1" noScreenShareCover)" "$3"
}
check ctl-default true "$WORK/cover.png"
check ctl-rule true "$WORK/rule \"1\".png"
check ctl-blur true "blur 24"
check ctl-pixelate true "pixelate 8"
check ctl-gone true "transparent"
check ctl-plain false ""
same "stableId is still there" "$(hctl clients -j | jq '[.[] | select(.stableId)] | length')" 6
same "the fields end their block" "$(hctl clients | grep -A1 'noScreenShareCover: ' | grep -c '^$')" 6

echo "== hyprctl activewindow"
active=$(hctl activewindow -j | jq -r .class)
same "noScreenShare (json)" "$(hctl activewindow -j | jq -r .noScreenShare)" "$(json "$active" noScreenShare)"
same "noScreenShareCover" "$(hctl activewindow | sed -n 's/^\tnoScreenShareCover: //p')" "$(plain "$active" noScreenShareCover)"
same "batch" "$(hctl --batch 'j/clients ; j/activewindow' | grep -c '"noScreenShare":')" 7

echo "== unloaded"
hctl plugin unload "$PLUGIN" >/dev/null; sleep 1
same "fields in clients" "$(hctl clients | grep -c noScreenShare)" 0
same "fields in clients -j" "$(hctl clients -j | jq '[.[] | select(has("noScreenShare"))] | length')" 0
hctl plugin load "$PLUGIN" >/dev/null; sleep 1
same "loaded again: fields once per window" "$(hctl clients | grep -c 'noScreenShare: ')" 6

echo "== another plugin wrapping clients"
# It puts a lambda of its own in place of the handler it finds (a line before the answer),
# and puts that handler back on unload. Built like this plugin (-fno-gnu-unique), so its
# code is really gone once it is unloaded. Two of them:
#   blind:   the common way, it keeps a copy of what it found and puts it back whatever is
#            there by then;
#   careful: the way this plugin does it, only its wrapper holds what it found, put back
#            only while a wrapper of its own is still held.
cat > "$WORK/other.cpp" <<'CPP'
#include "helpers/defer/Promise.hpp"
#include "helpers/memory/Memory.hpp"
#define private public
#include "ipc/s1/S1.hpp"
#undef private
#include "plugins/PluginAPI.hpp"

namespace {
    using Handler = std::function<IPC::Socket1::SResponse(const IPC::Socket1::SRequest&)>;
    SP<IPC::Socket1::SCommand> g_cmd;
#ifdef BLIND
    Handler g_found;
#else
    std::weak_ptr<const Handler> g_found;
#endif
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE) {
    for (const auto& cmd : IPC::Socket1::sock()->m_commands) {
        if (!cmd || cmd->name != "clients" || cmd->match != IPC::Socket1::COMMAND_MATCH_EXACT)
            continue;
        auto found = std::make_shared<const Handler>(cmd->handler);
        g_cmd      = cmd;
#ifdef BLIND
        g_found = *found;
#else
        g_found = found;
#endif
        cmd->handler = [found](const IPC::Socket1::SRequest& request) {
            auto response = (*found)(request);
            if (auto* text = std::get_if<std::string>(&response.result); text && request.format == IPC::Socket1::FORMAT_NORMAL)
                *text = "wrapped-by-other\n" + *text;
            return response;
        };
    }
    return {"ctl-other", "wraps hyprctl clients", "test", "1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
#ifdef BLIND
    if (g_cmd)
        g_cmd->handler = std::move(g_found);
#else
    if (const auto found = g_found.lock(); g_cmd && found)
        g_cmd->handler = *found;
#endif
    g_cmd.reset();
}
CPP
BLIND=$WORK/libctl-blind.so
CAREFUL=$WORK/libctl-careful.so
build_other() { # $1: output, $2...: extra flags
    c++ -std=c++26 -shared -fPIC -fno-gnu-unique -O2 "${@:2}" \
        $(pkg-config --cflags hyprland pixman-1 libdrm wayland-server egl hyprutils hyprgraphics aquamarine hyprlang) \
        -o "$1" "$WORK/other.cpp" 2> "$WORK/other.log" || { fail "can't build $1"; tail -5 "$WORK/other.log"; }
}
build_other "$BLIND" -DBLIND
build_other "$CAREFUL"

load() { hctl plugin load "$1" >/dev/null; sleep 1; }
unload() { hctl plugin unload "$1" >/dev/null; sleep 1; }
# Hyprland still runs and answers with every window
alive() { hctl clients -j 2>/dev/null | jq -e 'length == 6' >/dev/null 2>&1; }
marks() { hctl clients | grep -c '^wrapped-by-other$'; }
fields() { hctl clients | grep -c 'noScreenShare: '; }

if [ -f "$BLIND" ] && [ -f "$CAREFUL" ]; then
    unload "$PLUGIN"

    echo "-- blind, loaded before the plugin and unloaded before it"
    load "$BLIND"
    load "$PLUGIN"
    same "both answer: its line" "$(marks)" 1
    same "both answer: the fields" "$(fields)" 6
    # it puts back Hyprland's handler, and our wrapper (holding its own) goes
    unload "$BLIND"
    alive && pass "it unloaded first" || fail "Hyprland died as the other plugin unloaded"
    same "its line is gone" "$(marks)" 0
    same "the fields went with it" "$(fields)" 0
    # nothing of its code is left here to call or destroy
    unload "$PLUGIN"
    alive && pass "the plugin unloaded after it: Hyprland lives" || fail "Hyprland died as the plugin unloaded after the other one"
    alive && same "clients is Hyprland's own" "$(marks)/$(fields)" 0/0

    echo "-- blind, loaded before the plugin and unloaded after it"
    load "$BLIND"
    load "$PLUGIN"
    unload "$PLUGIN"
    alive && pass "the plugin unloaded first: Hyprland lives" || fail "Hyprland died as the plugin unloaded before the other one"
    alive && same "its line stays, the fields are gone" "$(marks)/$(fields)" 1/0
    unload "$BLIND"
    alive && pass "it unloaded after: Hyprland lives" || fail "Hyprland died as the other plugin unloaded after"
    alive && same "clients is Hyprland's own" "$(marks)/$(fields)" 0/0

    # Wrapped around the plugin's wrapper, the other one goes with it when the plugin
    # unloads first: nothing can be taken out of the middle of the chain, and kept, it would
    # call into the unloaded plugin. (A blind one would then put back the plugin's wrapper
    # on its own unload, code of an unloaded plugin: that one is not tried.)
    echo "-- careful, loaded after the plugin and unloaded after it"
    load "$PLUGIN"
    load "$CAREFUL"
    same "both answer: its line" "$(marks)" 1
    same "both answer: the fields" "$(fields)" 6
    unload "$PLUGIN"
    alive && pass "the plugin unloaded first: Hyprland lives" || fail "Hyprland died as the plugin unloaded under the other one"
    alive && same "both wrappers are gone" "$(marks)/$(fields)" 0/0
    unload "$CAREFUL"
    alive && pass "it unloaded after: Hyprland lives" || fail "Hyprland died as the other plugin unloaded after"
    alive && same "clients is Hyprland's own" "$(marks)/$(fields)" 0/0

    echo "-- careful, loaded after the plugin and unloaded before it"
    load "$PLUGIN"
    load "$CAREFUL"
    unload "$CAREFUL"
    alive && pass "it unloaded first: Hyprland lives" || fail "Hyprland died as the other plugin unloaded over the plugin"
    alive && same "its line is gone, the fields stay" "$(marks)/$(fields)" 0/6
    unload "$PLUGIN"
    alive && pass "the plugin unloaded after it: Hyprland lives" || fail "Hyprland died as the plugin unloaded after the other one"
    alive && same "clients is Hyprland's own" "$(marks)/$(fields)" 0/0
fi

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed"
exit "$FAILS"
