#!/usr/bin/env bash
# Covers in Hyprland's stacking order:
#   - an opaque window with rounded corners on top of a hidden one keeps its corners: the
#     cover goes only where the hidden window shows, outside the rounding; in a region
#     capture (where nothing is drawn rounded) the hidden window never shows in its corners;
#   - and its border, edges and rounded corners, which lie outside its box over the hidden
#     window: the cover goes only outside the border;
#   - layers: a top layer that isn't hidden stays under a hidden overlay layer's cover, an
#     overlay layer that isn't hidden stays over a hidden top layer's cover;
#   - a closing tiled window's cover stays under a floating window above it (Hyprland draws
#     the fade-out of a tiled window under the floating ones), and still hides the rest;
#   - the same for its close_hold without a close animation (no fade-out to place it by).
# grim is the capture client, the cover is plain magenta.
#
#   tests/e2e/stacking.sh <libnoshare-cover.so>
#
# Requires: Hyprland 0.57+, grim, foot, jq, imagemagick; a seat; a monitor of 1280x720 or
# more. The hold part needs wf-recorder, the layer part quickshell. Exit code 0 means all
# passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-stack.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export NOSHARE_COVER_DEBUG=$WORK/trace.log
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

magick -size 640x360 xc:'#ff00ff' "$WORK/cover.png"
R=30 # rounding

write_config() { # $1: animations (true/false), $2: close_hold ms (default 0), $3: border size (default 0)
    cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/cover.png", close_hold = ${2:-0} } },
    general = {
        gaps_in = 0, gaps_out = 0, border_size = ${3:-0}, layout = "dwindle",
        col = { active_border = "rgba(0000ffff)", inactive_border = "rgba(0000ffff)" },
    },
    decoration = { rounding = $R, shadow = { enabled = false }, blur = { enabled = false } },
    animations = { enabled = $1 },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true },
    debug = { suppress_errors = true },
})
-- only closing is animated, slowly enough to take shots during it (3 s)
hl.animation({ leaf = "windowsIn", enabled = false })
hl.animation({ leaf = "fadeIn", enabled = false })
hl.animation({ leaf = "windowsMove", enabled = false })
hl.animation({ leaf = "layers", enabled = false })
hl.animation({ leaf = "fadeLayers", enabled = false })
hl.animation({ leaf = "windowsOut", enabled = true, speed = 30, bezier = "default", style = "popin 50%" })
hl.animation({ leaf = "fadeOut", enabled = true, speed = 30, bezier = "default" })
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
hl.window_rule({ match = { class = "cover-float" }, no_screen_share = true, float = true, size = { 500, 300 }, move = { 700, 350 } })
hl.window_rule({ match = { class = "over" }, float = true, size = { 600, 400 }, move = { 200, 150 } })
hl.layer_rule({ match = { namespace = "nsc-top-hidden" }, no_screen_share = true })
hl.layer_rule({ match = { namespace = "nsc-over-hidden" }, no_screen_share = true })
LUA
}

hctl() { hyprctl -i 0 "$@"; }
shot() { grim "$WORK/$1.png"; }
win_box() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }
win_pid() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | .pid' | head -1; }
mean_rgb() { magick "$WORK/$1.png" -crop "$2" -resize 1x1\! -format '%[fx:int(255*r)] %[fx:int(255*g)] %[fx:int(255*b)]' info:; }
is_cover() { read -r r g b <<<"$1"; [ "$r" -gt 200 ] && [ "$g" -lt 60 ] && [ "$b" -gt 200 ]; }
is_green() { read -r r g b <<<"$1"; [ "$r" -lt 60 ] && [ "$g" -gt 200 ] && [ "$b" -lt 60 ]; }
is_yellow() { read -r r g b <<<"$1"; [ "$r" -gt 200 ] && [ "$g" -gt 200 ] && [ "$b" -lt 60 ]; }
is_blue() { read -r r g b <<<"$1"; [ "$r" -lt 60 ] && [ "$g" -lt 60 ] && [ "$b" -gt 200 ]; }
is_cover_or_green() { is_cover "$1" || is_green "$1"; }
expect() { # $1 shot, $2 crop, $3 cover|green|yellow|blue|cover_or_green, $4 label
    local c; c=$(mean_rgb "$1" "$2")
    if "is_$3" "$c"; then pass "$4 ($c)"; else fail "$4: want $3, got $c ($WORK/$1.png, $2)"; fi
}

start_hyprland() {
    pkill -xu "$(id -u)" Hyprland; sleep 0.5
    Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
    for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
    export WAYLAND_DISPLAY=$(ls -t "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
    sleep 2
    read -r MW MH <<<"$(hctl monitors -j | jq -r '.[0] | "\(.width) \(.height)"')"
    if [ "${MW:-0}" -lt 1280 ] || [ "${MH:-0}" -lt 720 ]; then
        fail "the monitor is ${MW:-?}x${MH:-?}, the test needs 1280x720 or more"
        pkill -xu "$(id -u)" Hyprland
        exit 1
    fi
}

open_win() { # class [foot args...]: a static window, no cursor
    local class=$1; shift
    foot --app-id "$class" "$@" sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
    for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box "$class")" ] && break; done
    sleep 1
}

# The terminal must mark its surface opaque (foot and kitty do at full opacity): a translucent
# window is drawn again over the whole cover anyway.
echo "== rounded corners of an opaque window on top of a hidden one"
write_config false
start_hyprland
open_win cover-me
open_win over -o colors-dark.background=00ff00
[ "$(win_box over)" = "600x400+200+150" ] || fail "over is at $(win_box over), not 600x400+200+150"
shot corners
# in each corner: inside the rounding (10..14 px in, both ways) the window, at its very
# corner (outside the rounding) the hidden window's cover
for corner in "200 150 1 1" "800 150 -1 1" "200 550 1 -1" "800 550 -1 -1"; do
    read -r cx cy sx sy <<<"$corner"
    ix=$((cx + sx * 12 - (sx < 0 ? 4 : 0))); iy=$((cy + sy * 12 - (sy < 0 ? 4 : 0)))
    ox=$((cx + sx * 1 - (sx < 0 ? 2 : 0))); oy=$((cy + sy * 1 - (sy < 0 ? 2 : 0)))
    expect corners "4x4+$ix+$iy" green "corner at $cx,$cy: the window inside its rounding"
    expect corners "2x2+$ox+$oy" cover "corner at $cx,$cy: the cover outside its rounding"
done
expect corners "100x100+450+300" green "the window's middle"
# A region capture copies the monitor's picture, where the window is rounded, but draws its
# covers and redraws square: the cover must still go into the corners (and the window over
# it there, square), never leave the hidden window showing outside the rounding.
RX=150; RY=100
grim -g "$RX,$RY 800x600" "$WORK/corners-region.png"
for corner in "200 150 1 1" "800 150 -1 1" "200 550 1 -1" "800 550 -1 -1"; do
    read -r cx cy sx sy <<<"$corner"
    ix=$((cx + sx * 12 - (sx < 0 ? 4 : 0) - RX)); iy=$((cy + sy * 12 - (sy < 0 ? 4 : 0) - RY))
    ox=$((cx + sx * 1 - (sx < 0 ? 2 : 0) - RX)); oy=$((cy + sy * 1 - (sy < 0 ? 2 : 0) - RY))
    expect corners-region "4x4+$ix+$iy" green "region capture, corner at $cx,$cy: the window inside its rounding"
    expect corners-region "2x2+$ox+$oy" cover_or_green "region capture, corner at $cx,$cy: no hidden window outside its rounding"
done
expect corners-region "100x100+$((450 - RX))+$((300 - RY))" green "region capture: the window's middle"
expect corners-region "40x40+$((850 - RX))+$((300 - RY))" cover "region capture: the hidden window next to it is covered"

# The border lies outside the window's box, so the cover went over it, rounded corners and
# all, and nothing drew it again. A plain blue border, the same focused or not.
echo "== the border of an opaque window on top of a hidden one"
B=8
write_config false 0 "$B"
hctl reload >/dev/null; sleep 1.5
read -r W H X Y <<<"$(win_box over | tr 'x+' '  ')"
[ "${W:-0}" -ge 400 ] || fail "over is at $(win_box over)"
shot border
expect border "2x20+$((X - B / 2 - 1))+$((Y + H / 2 - 10))" blue "the left edge of the border"
expect border "2x20+$((X + W + B / 2 - 1))+$((Y + H / 2 - 10))" blue "the right edge of the border"
expect border "20x2+$((X + W / 2 - 10))+$((Y - B / 2 - 1))" blue "the top edge of the border"
expect border "20x2+$((X + W / 2 - 10))+$((Y + H + B / 2 - 1))" blue "the bottom edge of the border"
expect border "2x20+$((X - B - 6))+$((Y + H / 2 - 10))" cover "the hidden window outside the border is covered"
# each rounded corner: its middle (45 degrees, halfway through the ring) the border, outside
# the ring the cover. The ring runs from R to R + B around the point R in from the corner.
O=$(((R + B / 2) * 707 / 1000))
for corner in "$X $Y 1 1" "$((X + W)) $Y -1 1" "$X $((Y + H)) 1 -1" "$((X + W)) $((Y + H)) -1 -1"; do
    read -r cx cy sx sy <<<"$corner"
    mx=$((cx + sx * (R - O) - 1)); my=$((cy + sy * (R - O) - 1))
    ox=$((cx - sx * (B - 2) - (sx < 0 ? 2 : 0))); oy=$((cy - sy * (B - 2) - (sy < 0 ? 2 : 0)))
    expect border "2x2+$mx+$my" blue "corner at $cx,$cy: the border's rounding"
    expect border "2x2+$ox+$oy" cover "corner at $cx,$cy: the cover outside the border's rounding"
done
expect border "100x100+$((X + W / 2 - 50))+$((Y + H / 2 - 50))" green "the window's middle, with a border"
grep -aq "border drawn again over a cover" "$WORK/trace.log" && pass "the border was drawn again (trace)" ||
    fail "the border was never drawn again ($WORK/trace.log)"
# a region capture draws it square, the edges the same
grim -g "$RX,$RY 800x600" "$WORK/border-region.png"
expect border-region "2x20+$((X - B / 2 - 1 - RX))+$((Y + H / 2 - 10 - RY))" blue "region capture: the left edge of the border"
expect border-region "20x2+$((X + W / 2 - 10 - RX))+$((Y - B / 2 - 1 - RY))" blue "region capture: the top edge of the border"

echo "== a closing tiled window's cover under a floating window"
write_config true
hctl reload >/dev/null; sleep 1.5
kill "$(win_pid cover-me)"
for _ in $(seq 100); do [ -z "$(win_box cover-me)" ] && break; sleep 0.01; done
sleep 0.5 # well inside the 3 s close
shot closing
expect closing "100x100+450+300" green "the floating window stays over the closing cover"
expect closing "100x100+$((MW / 2 + 200))+$((MH / 2 - 50))" cover "the rest of the closing window is still covered"
sleep 3

# Without a close animation Hyprland makes no fade-out (or one that is over at once): the
# cover can't go where a fade-out is drawn, it goes where the window was. Over every window,
# it would hide the floating one for the whole hold. A stream runs across the close: grim
# alone never sees the window close (one frame, long before).
if command -v wf-recorder >/dev/null; then
    echo "== a closed tiled window's hold under a floating window, no close animation"
    write_config false 3000
    hctl reload >/dev/null; sleep 1.5
    open_win cover-me
    wf-recorder -f "$WORK/hold.mkv" -c libx264 -p preset=ultrafast >/dev/null 2>&1 &
    REC=$!
    sleep 1.5
    kill "$(win_pid cover-me)"
    for _ in $(seq 100); do [ -z "$(win_box cover-me)" ] && break; sleep 0.01; done
    sleep 0.5 # well inside the 3 s hold
    shot hold
    kill "$REC" 2>/dev/null; wait "$REC" 2>/dev/null
    expect hold "100x100+450+300" green "the floating window stays over the closed window's hold"
    expect hold "100x100+$((MW / 2 + 200))+$((MH / 2 - 50))" cover "the rest of the closed window is still covered during its hold"
    sleep 3 # the hold is over before the next part
else
    echo "  no wf-recorder, the hold part is skipped"
fi

if command -v quickshell >/dev/null; then
    echo "== layers: top and overlay, hidden and not"
    kill "$(win_pid over)" 2>/dev/null
    write_config false
    hctl reload >/dev/null; sleep 1.5
    open_win plain
    open_win cover-float
    [ "$(win_box cover-float)" = "500x300+700+350" ] || fail "cover-float is at $(win_box cover-float), not 500x300+700+350"
    layer() { # $1 namespace, $2 Top|Overlay, $3 top margin, $4 left margin, $5 w, $6 h, $7 color
        mkdir -p "$WORK/$1"
        cat > "$WORK/$1/shell.qml" <<QML
import Quickshell
import Quickshell.Wayland
import QtQuick
PanelWindow {
    WlrLayershell.namespace: "$1"
    WlrLayershell.layer: WlrLayer.$2
    exclusionMode: ExclusionMode.Ignore
    anchors { top: true; left: true }
    margins { top: $3; left: $4 }
    implicitWidth: $5
    implicitHeight: $6
    color: "$7"
}
QML
        quickshell -p "$WORK/$1/shell.qml" > "$WORK/$1.log" 2>&1 &
        QS+=($!)
        for _ in $(seq 40); do sleep 0.25; hctl layers -j | grep -q "\"$1\"" && break; done
    }
    QS=()
    # over the plain window: a hidden top layer, a visible overlay layer over its right half
    layer nsc-top-hidden Top 80 60 400 200 "#0000ff"
    layer nsc-over Overlay 80 260 400 200 "#00ff00"
    # over the hidden window: a hidden overlay layer, a visible top layer under its right part
    layer nsc-over-hidden Overlay 400 750 300 150 "#0000ff"
    layer nsc-top Top 450 900 300 150 "#ffff00"
    sleep 1
    n=$(hctl layers -j | jq '[.. | objects | select(.namespace? | strings | startswith("nsc-"))] | length')
    [ "$n" -eq 4 ] || fail "only $n of the 4 quickshell layers mapped ($WORK/*.log)"
    shot layers
    expect layers "40x40+140+160" cover "the hidden top layer is covered"
    expect layers "40x40+340+160" green "the overlay layer stays over the hidden top layer's cover"
    expect layers "40x40+780+430" cover "the hidden overlay layer is covered"
    expect layers "40x40+955+480" cover "the hidden overlay layer's cover stays over the top layer"
    expect layers "40x40+1100+560" yellow "the top layer stays over the hidden window's cover"
    kill "${QS[@]}" 2>/dev/null
else
    echo "  no quickshell, the layer part is skipped"
fi

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed ($WORK)"
exit "$FAILS"
