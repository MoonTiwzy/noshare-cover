#!/usr/bin/env bash
# decoration:wobble bends a window while it moves. Its cover bends with it: a stream that
# watches a hidden window being thrown around never shows the window from behind the cover,
# and a window that wobbles above a hidden one doesn't uncover it either. wf-recorder is the
# capture client.
#
#   tests/e2e/wobble.sh <libnoshare-cover.so>
#
# Requires: Hyprland 0.57+, foot, wf-recorder, ffmpeg (libx264rgb), jq, imagemagick; a seat.
# Exit code 0 means all passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-wobble.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

# the hidden window is red, its cover green, the window above them blue
magick -size 64x64 xc:'#00ff00' "$WORK/cover.png"

cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/cover.png" } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    decoration = {
        rounding = 0, shadow = { enabled = false }, blur = { enabled = false },
        -- a long, wide wobble
        wobble = { enabled = true, intensity = 1.5, stiffness = 60, damping = 4 },
    },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true, disable_watchdog_warning = true },
    debug = { suppress_errors = true },
})
hl.window_rule({ match = { class = "wob-me" }, no_screen_share = true, float = true, size = "500 400", move = "200 200" })
hl.window_rule({ match = { class = "wob-over" }, float = true, size = "300 300", move = "1100 300" })
LUA

hctl() { hyprctl -i 0 "$@"; }
win_at() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.at[0]) \(.at[1])"' | head -1; }
# pixels of one color in a frame, give or take what the codec does at an edge
count() { magick "$1" -fuzz 20% -fill black +opaque "$2" -fill white -opaque "$2" -format '%[fx:int(mean*w*h+0.5)]' info:; }

pkill -xu "$(id -u)" Hyprland; sleep 0.5
Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
export WAYLAND_DISPLAY=$(ls "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
sleep 2

foot --app-id plain -o colors-dark.background=000000 sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_at plain)" ] && break; done
foot --app-id wob-me -o colors-dark.background=ff0000 sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_at wob-me)" ] && break; done
sleep 1

# $1: name, $2: class to throw around; records the stream meanwhile into $WORK/$1-NNN.png
throw() {
    wf-recorder -y -f "$WORK/$1.mkv" -c libx264rgb -p preset=ultrafast -p crf=0 >/dev/null 2>&1 &
    local rec=$!
    sleep 1.5
    hctl dispatch "hl.dsp.focus({ window = \"class:$2\" })" >/dev/null
    for step in "300 0" "0 250" "-300 0" "0 -250" "350 200" "-350 -200"; do
        hctl dispatch "hl.dsp.window.move({ x = ${step% *}, y = ${step#* }, relative = true })" >/dev/null
        sleep 0.35
    done
    sleep 1.5 # until it stands still again
    kill -INT "$rec"; wait "$rec" 2>/dev/null
    ffmpeg -loglevel error -i "$WORK/$1.mkv" -vf fps=30 "$WORK/$1-%03d.png"
}

# over the frames of $1: the most red seen in one, the frames whose green isn't a rectangle
survey() {
    RED=0 BENT=0 FRAMES=0
    local f n box area
    for f in "$WORK/$1"-*.png; do
        [ -f "$f" ] || continue
        FRAMES=$((FRAMES + 1))
        n=$(count "$f" '#ff0000'); [ "$n" -gt "$RED" ] && RED=$n
        box=$(magick "$f" -fuzz 20% -fill black +opaque '#00ff00' -format '%@' info: 2>/dev/null) # WxH+X+Y of the green
        area=$(( ${box%%x*} * $(echo "$box" | sed 's/^[0-9]*x\([0-9]*\).*/\1/') ))
        n=$(count "$f" '#00ff00')
        [ "$area" -gt 0 ] && [ $((area - n)) -gt $((area / 100)) ] && BENT=$((BENT + 1))
    done
}

echo "== a hidden window that wobbles"
before=$(win_at wob-me)
throw hidden wob-me
survey hidden
[ "$FRAMES" -ge 30 ] && pass "the stream has $FRAMES frames" || fail "the stream has only $FRAMES frames"
[ "$(win_at wob-me)" = "$before" ] && pass "the window went around and came back" || fail "the window didn't move as told ($before -> $(win_at wob-me))"
[ "$BENT" -ge 5 ] && pass "the cover bends with the window ($BENT frames)" || fail "the cover stays a rectangle ($BENT bent frames): no wobble, or the cover doesn't follow"
[ "$RED" -le 4 ] && pass "the window never shows from behind the cover (at most $RED px)" || fail "the window shows from behind the cover: $RED px in one frame"

echo "== a window that wobbles above a hidden one"
foot --app-id wob-over -o colors-dark.background=0000ff sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_at wob-over)" ] && break; done
# over the hidden window, so it sweeps across it
hctl dispatch 'hl.dsp.focus({ window = "class:wob-over" })' >/dev/null
hctl dispatch 'hl.dsp.window.move({ x = 350, y = 250 })' >/dev/null
sleep 2
throw over wob-over
survey over
[ "$RED" -le 4 ] && pass "the hidden window never shows around it (at most $RED px)" || fail "the hidden window shows around it: $RED px in one frame"
blue=$(count "$WORK/over-001.png" '#0000ff')
[ "$blue" -gt 60000 ] && pass "it stays on top of the cover ($blue px of it)" || fail "it is lost under the cover ($blue px of it)"
least=$blue
for f in "$WORK"/over-*.png; do n=$(count "$f" '#0000ff'); [ "$n" -lt "$least" ] && least=$n; done
[ "$least" -gt $((blue * 8 / 10)) ] && pass "also while it wobbles (at least $least px)" || fail "it loses its picture while it wobbles ($least of $blue px)"

pkill -xu "$(id -u)" Hyprland
echo "(frames in $WORK)"
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed"
exit "$FAILS"
