#!/usr/bin/env bash
# noshare-cover + gloview together on a live Hyprland: load order, unload/reload, overview tiles.
#   tests/e2e/with-gloview.sh <libnoshare-cover.so> <gloview.so>
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
NSC=$(realpath "${1:?libnoshare-cover.so}")
GLV=$(realpath "${2:?gloview.so}")
W=$(mktemp -d /tmp/nsc-glv.XXXXXX); 
magick -size 64x64 xc:'#ff00ff' $W/cover.png
FAILS=0
pass() { printf '  ok   %s\n' "$*"; }
fail() { printf '  FAIL %s\n' "$*"; FAILS=$((FAILS + 1)); }
hctl() { hyprctl -i 0 "$@"; }
box() { hctl clients -j | jq -r '.[] | select(.class == "cover-me") | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }
magenta_share() { # fraction of pure-ish magenta pixels in the whole capture
    grim $W/s.png
    magick $W/s.png -fx 'r>0.85 && g<0.2 && b>0.85 ? 1 : 0' -format '%[fx:mean]' info:
}
win_is_cover() { grim $W/w.png; magick $W/w.png -crop "$(box)" -resize 1x1\! -format '%[fx:int(255*r)] %[fx:int(255*g)] %[fx:int(255*b)]' info:; }
plugins() { hctl plugin list | grep -o 'Plugin [A-Za-z-]*' | tr '\n' ' '; }

start() {
    pkill -x Hyprland; sleep 0.7
    cat > $W/h.lua <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$W/cover.png" } },
    debug = { suppress_errors = true },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true },
})
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
LUA
    Hyprland --config $W/h.lua > $W/hypr.log 2>&1 &
    for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
    export WAYLAND_DISPLAY=$(ls $XDG_RUNTIME_DIR | grep -m1 '^wayland-[0-9]*$')
    sleep 1
    foot --app-id cover-me sh -c 'sleep 600' >/dev/null 2>&1 &
    foot --app-id plain sh -c 'sleep 600' >/dev/null 2>&1 &
    sleep 2.5
}
load() { hctl plugin load "$1" >/dev/null; hctl reload >/dev/null; sleep 1.5; }
unload() { hctl plugin unload "$1" >/dev/null; hctl reload >/dev/null; sleep 1.5; }
check_cover() { # label
    local c; c=$(win_is_cover); read -r r g b <<<"$c"
    [ "$r" -gt 200 ] && [ "$g" -lt 60 ] && [ "$b" -gt 200 ] && pass "$1: window is covered in the capture" || fail "$1: capture does not show the cover ($c) [$(plugins)]"
}

echo "== 1. gloview first, noshare-cover second"
start; load $GLV; load $NSC
echo "     plugins: $(plugins)"
check_cover "gloview→noshare"

echo "== 2. unload noshare-cover while gloview is loaded"
unload $NSC
[ -n "$(pgrep -x Hyprland)" ] && pass "Hyprland alive" || fail "Hyprland crashed"
[ "$(grep -c noshare /proc/$(pgrep -x Hyprland)/maps)" -eq 0 ] && pass "noshare-cover unmapped from memory (gloview holds no handle)" || fail "noshare-cover still in memory"
c=$(win_is_cover); read -r r g b <<<"$c"
[ "$r" -lt 10 ] && [ "$g" -lt 10 ] && [ "$b" -lt 10 ] && pass "without noshare-cover the window is black in the capture (Hyprland)" || fail "without noshare-cover: $c"

echo "== 3. load noshare-cover again"
load $NSC
check_cover "reload"

echo "== 4. gloview overlay open: window preview in the capture = cover"
hctl gloview >/dev/null 2>&1 || hctl dispatch 'hl.dsp.exec_cmd("true")' >/dev/null
sleep 1.5
f1=$(magenta_share)
hctl gloview >/dev/null 2>&1; sleep 1.2
awk "BEGIN{exit !($f1 > 0.005)}" && pass "preview shows the cover in the capture (magenta share $f1)" || fail "no cover on the overlay preview (share $f1)"
grep -q "Path C" $W/hypr.log "$XDG_RUNTIME_DIR"/hypr/*/hyprland.log 2>/dev/null; true

echo "== 5. noshare-cover first, gloview second"
start; load $NSC; load $GLV
echo "     plugins: $(plugins)"
check_cover "noshare→gloview"

echo "== 6. 5 unload/load cycles of both in varying order"
for i in 1 2 3 4 5; do unload $GLV; unload $NSC; load $NSC; load $GLV; unload $NSC; load $NSC; done
[ -n "$(pgrep -x Hyprland)" ] && pass "Hyprland alive after cycles" || fail "Hyprland crashed"
check_cover "after cycles"
pkill -x Hyprland
echo; [ $FAILS -eq 0 ] && echo "ALL PASSED" || echo "FAILURES: $FAILS"
