#!/usr/bin/env bash
# `hyprctl clients` and `hyprctl activewindow` show what the plugin does to each window
# (noScreenShare, noScreenShareCover), in both formats, and are Hyprland's own again once
# the plugin is unloaded.
#
#   tests/e2e/hyprctl.sh <libnoshare-cover.so>
#
# Requires: Hyprland, foot, jq, imagemagick; a seat. Exit code 0 means all passed.
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

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed"
exit "$FAILS"
