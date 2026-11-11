#!/usr/bin/env bash
# Live check of the two cases Hyprland doesn't cover by itself:
#   - a closing no_screen_share window (the close animation is a snapshot that
#     no_screen_share doesn't apply to), with and without close_hold;
#   - cursor zoom (the screencast gets the zoomed image, the boxes must follow).
#
#   tests/e2e/close-zoom.sh <libnoshare-cover.so>
#
# Same requirements as run.sh, plus wf-recorder and slurp. Exit code 0 means all passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-cz.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export NOSHARE_COVER_DEBUG=$WORK/trace.log
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

magick -size 640x360 xc:'#ff00ff' "$WORK/cover.png"

write_config() { # $1: close_hold ms, $2: close animation speed (ds, 0 = off), $3: zoom factor, $4: blur (true/false)
    local anim="animations = { enabled = false },"
    [ "$2" != 0 ] && anim="animations = { enabled = true },"
    cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/cover.png", close_hold = $1 } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    decoration = { rounding = 0, shadow = { enabled = false }, blur = { enabled = ${4:-false} } },
    cursor = { zoom_factor = $3 },
    $anim
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true },
    debug = { suppress_errors = true },
})
hl.curve("linear", { type = "bezier", points = { {0, 0}, {1, 1} } })
hl.animation({ leaf = "windowsOut", enabled = true, speed = ${2/#0/1}, bezier = "linear", style = "popin 87%" })
hl.animation({ leaf = "fadeOut", enabled = true, speed = ${2/#0/1}, bezier = "linear" })
hl.animation({ leaf = "windowsIn", enabled = false })
hl.animation({ leaf = "fadeIn", enabled = false })
hl.animation({ leaf = "windowsMove", enabled = false })
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
LUA
}

hctl() { hyprctl -i 0 "$@"; }
shot() { grim "$WORK/$1.png"; }
win_box() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }
win_pid() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | .pid' | head -1; }
mean_rgb() { magick "$WORK/$1.png" -crop "$2" -resize 1x1\! -format '%[fx:int(255*r)] %[fx:int(255*g)] %[fx:int(255*b)]' info:; }
is_cover() { read -r r g b <<<"$1"; [ "$r" -gt 200 ] && [ "$g" -lt 60 ] && [ "$b" -gt 200 ]; }
is_black() { read -r r g b <<<"$1"; [ "$r" -lt 8 ] && [ "$g" -lt 8 ] && [ "$b" -lt 8 ]; }
hypr_pid() { pgrep -xu "$(id -u)" Hyprland | head -1; }
# central part of a WxH+X+Y box, $2 percent of each side
inner() { local w h x y; IFS='x+' read -r w h x y <<<"$1"; local p=$2
    echo "$((w * p / 100))x$((h * p / 100))+$((x + w * (100 - p) / 200))+$((y + h * (100 - p) / 200))"; }

start_hyprland() {
    pkill -xu "$(id -u)" Hyprland; sleep 0.5
    Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
    for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
    export WAYLAND_DISPLAY=$(ls -t "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
    sleep 2
}

open_win() { # class [static]
    if [ "${2:-}" = static ]; then
        foot --app-id "$1" sleep 1000 >/dev/null 2>&1 & # draws once, then nothing changes
    else
        foot --app-id "$1" sh -c 'while :; do date; sleep 0.2; done' >/dev/null 2>&1 &
    fi
    for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box "$1")" ] && break; done
    sleep 1
}

# A screencast keeps the output shared and asks for frames continuously; grim alone
# is a one-frame share. wf-recorder stands in for OBS/the portal.
STREAM=
stream_on() { wf-recorder -y -f "$WORK/stream.mkv" -c libx264 -p preset=ultrafast >/dev/null 2>&1 & STREAM=$!; sleep 1.5; }
# Hyprland 0.56.2 itself segfaults on exit if a screencast stopped less than ~0.5 s
# earlier (the session posts an IPC event after the event manager is gone), plugin
# or not, so give it a second before the next restart.
stream_off() { [ -n "$STREAM" ] && kill "$STREAM" 2>/dev/null; wait "$STREAM" 2>/dev/null; STREAM=; sleep 1; }

# $6: "quiet": one shot, then no capture at all for a while before closing;
#     "static": the stream runs but nothing on screen changes for a while before closing
close_case() { # $1 label, $2 close_hold, $3 anim speed (0 = no animation), $4 delay before the shot, $5 expect cover (1/0), $6 mode
    write_config "$2" "$3" 1
    start_hyprland
    local kind=; [ "${6:-}" = static ] && kind=static
    open_win cover-me $kind
    open_win plain $kind
    local box; box=$(inner "$(win_box cover-me)" 40)
    case "${6:-}" in
        quiet) shot before; sleep 2 ;;
        static)
            stream_on
            local f0; f0=$(grep -ac "frame: monitor" "$WORK/trace.log")
            sleep 2
            echo "     frames during the static 2 s: $(($(grep -ac "frame: monitor" "$WORK/trace.log") - f0))"
            ;;
        *) stream_on ;;
    esac
    kill "$(win_pid cover-me)"
    sleep "$4"
    shot "close-$1"
    stream_off
    local c; c=$(mean_rgb "close-$1" "$box")
    if [ "$5" = 1 ]; then
        is_cover "$c" && pass "$1: covered ($c)" || { fail "$1: window content in the stream ($c)"; grep -a "closed" "$WORK/trace.log" | tail -2; }
    else
        is_cover "$c" && fail "$1: cover still there ($c)" || pass "$1: cover gone ($c)"
    fi
}

echo "== closing window"
close_case "during the close animation" 0 30 0.5 1
close_case "close_hold keeps it" 1500 1 0.8 1
close_case "close_hold ends" 1500 1 2.8 0
close_case "closed after a static screen" 0 30 0.5 1 quiet
close_case "hold without animation, static stream" 3000 0 1.2 1 static

echo "== window over a hidden one"
write_config 0 0 1
start_hyprland
open_win cover-me
open_win plain
# plain floats, 400x300, in the middle of cover-me (the left half)
hctl dispatch 'hl.dsp.window.float({ action = "toggle" })' >/dev/null
hctl dispatch 'hl.dsp.window.resize({ x = 400, y = 300 })' >/dev/null
hctl dispatch 'hl.dsp.window.move({ x = 120, y = 250 })' >/dev/null
sleep 1
PB=$(win_box plain)
echo "     plain floating at $PB"
shot overlap
C=$(mean_rgb overlap "$(inner "$PB" 60)")
is_cover "$C" || is_black "$C" && fail "window on top is painted over ($C)" || pass "window on top stays visible ($C)"
# a strip of cover-me left of the floating window
IFS='x+' read -r pw ph px py <<<"$PB"
if [ "$px" -gt 60 ]; then
    C=$(mean_rgb overlap "40x$((ph / 2))+$((px - 50))+$((py + ph / 4))")
    is_cover "$C" && pass "hidden window around it is still covered ($C)" || fail "hidden window not covered next to the floating one ($C)"
fi
# A translucent window on top: what shows through it must be the cover (tinted by the
# window's own background), never the hidden window, and the window itself stays drawn.
# The hidden foot is dark gray: showing through, it would give ~36 36 36.
shows_cover_through() { read -r r g b <<<"$1"; [ "$r" -gt 70 ] && [ "$b" -gt 70 ] && [ "$g" -lt 70 ] && [ "$r" -lt 230 ]; }
glass_case() { # $1 label, $2 blur true/false, $3 class of the translucent window (cover-me: hidden itself)
    local cls=${3:-glass}
    write_config 0 0 1 "$2"
    start_hyprland
    open_win cover-me
    foot --app-id "$cls" -o colors-dark.alpha=0.6 sh -c 'while :; do date; sleep 0.2; done' >/dev/null 2>&1 &
    for _ in $(seq 40); do sleep 0.25; [ "$(hctl clients -j | jq --arg c "$cls" '[.[] | select(.class == $c)] | length')" -ge "$([ "$cls" = cover-me ] && echo 2 || echo 1)" ] && break; done
    sleep 1
    hctl dispatch 'hl.dsp.window.float({ action = "toggle" })' >/dev/null
    hctl dispatch 'hl.dsp.window.resize({ x = 400, y = 300 })' >/dev/null
    hctl dispatch 'hl.dsp.window.move({ x = 120, y = 250 })' >/dev/null
    sleep 1
    shot "glass-$2-$cls"
    # the active (last opened, floating) window of that class
    local box; box=$(hctl activewindow -j | jq -r '"\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"')
    local c; c=$(mean_rgb "glass-$2-$cls" "$(inner "$box" 60)")
    if [ "$cls" = cover-me ]; then
        # hidden itself: its own content must never be drawn, only the cover
        is_cover "$c" && pass "$1: covered, its content is not drawn ($c)" || fail "$1: its content leaks ($c)"
    else
        shows_cover_through "$c" && pass "$1: the cover shows through it ($c)" || fail "$1: expected the cover through the window, got $c"
    fi
}
kill "$(win_pid plain)"
glass_case "translucent window on top" false
glass_case "translucent window with blur on top" true
glass_case "hidden translucent window over a hidden one" true cover-me

# A blurred translucent window half over a hidden window and half over a normal one:
# the redraw (with blur) must stay on the hidden half. The normal window shows black
# text on white; if the blur spilled over, that half would turn into flat gray.
write_config 0 0 1 true
start_hyprland
open_win cover-me
foot --app-id plain -o colors-dark.background=ffffff -o colors-dark.foreground=000000 sh -c 'while :; do seq 1 3000 | tr "
" " "; sleep 5; done' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box plain)" ] && break; done
foot --app-id glass -o colors-dark.alpha=0.4 sh -c 'while :; do date; sleep 0.2; done' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box glass)" ] && break; done
sleep 1
hctl dispatch 'hl.dsp.window.float({ action = "toggle" })' >/dev/null
hctl dispatch 'hl.dsp.window.resize({ x = 600, y = 400 })' >/dev/null
hctl dispatch 'hl.dsp.window.move({ x = 340, y = 200 })' >/dev/null
sleep 1.5
shot straddle
# right part of the glass window (x 700..900), over the normal window
SD=$(magick "$WORK/straddle.png" -crop 200x250+700+300 -colorspace gray -format '%[fx:standard_deviation]' info:)
awk "BEGIN{exit !($SD > 0.08)}" && pass "blur stays on the hidden half, the normal window keeps its text (stddev $SD)" || fail "blur spilled over the normal window (stddev $SD)"
C=$(mean_rgb straddle "200x250+400+300")
shows_cover_through "$C" && pass "the hidden half shows the cover through the glass ($C)" || fail "hidden half: expected the cover through the glass, got $C"

# A layer on top of a hidden window (bar, launcher, notification) must stay visible:
# slurp puts a 50% black overlay layer over the whole screen.
if command -v slurp >/dev/null; then
    write_config 0 0 1
    start_hyprland
    open_win cover-me
    slurp -b '#00000080' >/dev/null 2>&1 &
    SL=$!
    sleep 1.5
    shot layer-on-top
    kill $SL 2>/dev/null
    C=$(mean_rgb layer-on-top "$(inner "$(win_box cover-me)" 50)")
    read -r r g b <<<"$C"
    [ "$r" -gt 90 ] && [ "$r" -lt 170 ] && [ "$b" -gt 90 ] && [ "$b" -lt 170 ] && [ "$g" -lt 40 ]         && pass "overlay layer on top of a hidden window stays visible ($C)" || fail "overlay layer hidden under the cover ($C)"
else
    echo "  (no slurp, skipping the layer case)"
fi

echo "== cursor zoom"
write_config 0 0 2
start_hyprland
grep -aq "zoom hook: on" "$WORK/trace.log" && pass "zoom hook installed" || fail "zoom hook not installed"
open_win cover-me
open_win plain
read -r MW MH < <(hctl monitors -j | jq -r '.[0] | "\(.width) \(.height)"')
# zoom anchored near the left edge, so the covered (left) window's edge moves far
# right of where it is unzoomed
hctl dispatch "hl.dsp.cursor.move({ x = $((MW / 10)), y = $((MH / 2)) })" >/dev/null
stream_on
sleep 1.5
shot zoom
stream_off
# where the covered window's right edge really is in the zoomed image, from the zoom
# box the plugin recorded (the anchor Hyprland uses isn't exactly the cursor)
IFS=', x' read -r ZX _ ZW _ < <(grep -a "zoom Virtual" "$WORK/trace.log" | tail -1 | sed 's/.*: //; s/ of.*//')
EDGE_UNZOOMED=$(IFS='x+' read -r w h x y <<<"$(win_box cover-me)"; echo $((x + w)))
EDGE=$(awk -v zx="$ZX" -v zw="$ZW" -v mw="$MW" -v e="$EDGE_UNZOOMED" 'BEGIN{printf "%d", zx + e * zw / mw}')
echo "     covered window edge: ${EDGE_UNZOOMED}px unzoomed, ${EDGE}px zoomed"
if [ "$EDGE" -gt $((EDGE_UNZOOMED + 120)) ]; then
    # between the unzoomed and the zoomed edge: the old code showed the window here
    C=$(mean_rgb zoom "$((EDGE - EDGE_UNZOOMED - 60))x$((MH / 2))+$((EDGE_UNZOOMED + 30))+$((MH / 4))")
    is_cover "$C" && pass "zoomed window is covered where it really is ($C)" || fail "zoomed window not covered past its unzoomed edge ($C)"
else
    fail "zoom too small to tell ($EDGE vs $EDGE_UNZOOMED)"
fi
if [ "$EDGE" -lt $((MW - 80)) ]; then
    P=$(mean_rgb zoom "60x$((MH / 2))+$((EDGE + 10))+$((MH / 4))")
    is_cover "$P" || is_black "$P" && fail "plain window hidden under zoom ($P)" || pass "plain window visible under zoom ($P)"
fi

pkill -xu "$(id -u)" Hyprland
grep -aiq "segfault\|crash" "$WORK/hypr.log" && fail "crash in Hyprland log" || pass "no crashes"
echo
[ "$FAILS" -eq 0 ] && echo "ALL PASSED ($WORK)" || echo "$FAILS FAILED ($WORK)"
exit "$FAILS"
