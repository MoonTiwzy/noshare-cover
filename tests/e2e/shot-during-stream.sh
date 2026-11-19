#!/usr/bin/env bash
# show_to while a stream is running: a region screenshot from an allowed client (a copy of
# grim named grim-share) shows the hidden window, the stream (wf-recorder) keeps the cover
# during and after the shot. An animated cover on a still screen keeps playing in the stream
# after such a shot. The other way round: a screenshot that gets the animated cover, between
# two streams that see the window as it is (copies of wf-recorder named wf-share), has the
# still screen repainted only for itself, not at the cover's pace for the streams after it.
#
#   tests/e2e/shot-during-stream.sh <libnoshare-cover.so>
#
# Requires: Hyprland, grim, wf-recorder, ffmpeg and ffprobe, foot, jq, imagemagick; a seat.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-shot.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export NOSHARE_COVER_DEBUG=$WORK/trace.log
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

magick -size 640x360 gradient:'#ff00ff-#00ffff' "$WORK/cover.png"
# 4 frames a second, two shades that both pass is_cover
magick -delay 25 -size 64x64 xc:'#ff00ff' xc:'#c040ff' -loop 0 "$WORK/anim.gif"
# The ELF of a program, to copy under another name. nixpkgs wraps many programs in a shell
# script (wrapProgram): a copy of the script would run .grim-wrapped, and the capture client
# wouldn't be called grim-share.
elf_of() { # $1 program
    local p n
    p=$(readlink -f "$(type -P "$1")") || return 1
    for n in 1 2 3; do
        [ "$(head -c4 "$p" 2>/dev/null)" = $'\x7fELF' ] && { echo "$p"; return 0; }
        p="$(dirname "$p")/.$(basename "$p")-wrapped"
    done
    echo "$1 ($(type -P "$1")) is not an ELF binary and no .$1-wrapped ELF was found next to it" >&2
    return 1
}
GRIM_ELF=$(elf_of grim) || exit 1
cp "$GRIM_ELF" "$WORK/grim-share"
WFR_ELF=$(elf_of wf-recorder) || exit 1
cp "$WFR_ELF" "$WORK/wf-share"

cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/cover.png", show_to = "grim-share, wf-share" } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    decoration = { rounding = 0, shadow = { enabled = false }, blur = { enabled = false } },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true },
    debug = { suppress_errors = true },
})
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
hl.window_rule({ match = { class = "still-me" }, no_screen_share = true, no_screen_share_cover = "$WORK/anim.gif" })
LUA

hctl() { hyprctl -i 0 "$@"; }
win_box() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }
mean_rgb() { magick "$1" -crop "$2" -resize 1x1\! -format '%[fx:int(255*r)] %[fx:int(255*g)] %[fx:int(255*b)]' info:; }
is_cover() { read -r r g b <<<"$1"; [ "$r" -gt 100 ] && [ "$b" -gt 200 ]; }

pkill -xu "$(id -u)" Hyprland; sleep 0.5
Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
export WAYLAND_DISPLAY=$(ls "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
sleep 2
foot --app-id cover-me sh -c 'while :; do date; sleep 0.2; done' >/dev/null 2>&1 &
WIN=$!
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box cover-me)" ] && break; done
sleep 1
BOX=$(win_box cover-me)
# region inside the window, like slurp would give (x,y WxH)
read -r W H X Y <<<"$(sed 's/[x+]/ /g' <<<"$BOX")"
REGION="$((X + 20)),$((Y + 20)) $((W - 40))x$((H - 40))"

echo "== screenshot during a stream"
wf-recorder -f "$WORK/stream.mkv" -c libx264 -r 10 >/dev/null 2>&1 &
REC=$!
sleep 2
"$WORK/grim-share" -g "$REGION" "$WORK/shot.png"
sleep 2
grim "$WORK/plain.png"
sleep 1
kill -INT "$REC"; wait "$REC" 2>/dev/null
sleep 0.6   # Hyprland 0.56.2 can crash on exit right after a screencast stops

C=$(mean_rgb "$WORK/shot.png" "$((W - 40))x$((H - 40))+0+0")
is_cover "$C" && fail "grim-share region shot shows the cover ($C)" || pass "grim-share region shot shows the window ($C)"
C=$(mean_rgb "$WORK/plain.png" "$BOX")
is_cover "$C" && pass "plain grim still gets the cover ($C)" || fail "plain grim sees the window ($C)"

ffmpeg -loglevel error -i "$WORK/stream.mkv" -vf fps=5 "$WORK/f%03d.png"
N=$(ls "$WORK"/f*.png 2>/dev/null | wc -l)
BAD=0
for f in "$WORK"/f*.png; do
    C=$(mean_rgb "$f" "$BOX"); is_cover "$C" || BAD=$((BAD + 1))
done
[ "$N" -gt 5 ] && [ "$BAD" -eq 0 ] && pass "stream kept the cover in all $N frames" || fail "stream: $BAD of $N frames show the window"

# Nothing on screen changes by itself: only the GIF cover has the monitor repainted, at its
# pace. A grim-share frame doesn't draw that cover, and it used to stop the repaints for the
# stream too: the stream froze on the frame before the shot.
echo "== animated cover on a still screen, screenshot during a stream"
kill "$WIN"
foot --app-id still-me sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box still-me)" ] && break; done
sleep 1.5
BOX=$(win_box still-me)
read -r W H X Y <<<"$(sed 's/[x+]/ /g' <<<"$BOX")"
wf-recorder -f "$WORK/anim.mkv" -c libx264 >/dev/null 2>&1 &
REC=$!
sleep 2
"$WORK/grim-share" -g "$((X + 20)),$((Y + 20)) $((W - 40))x$((H - 40))" "$WORK/anim-shot.png"
sleep 4
kill -INT "$REC"; wait "$REC" 2>/dev/null
sleep 0.6
C=$(mean_rgb "$WORK/anim-shot.png" "$((W - 40))x$((H - 40))+0+0")
is_cover "$C" && fail "grim-share shot of the still window shows the cover ($C)" || pass "grim-share shot of the still window shows the window ($C)"
# frames of the stream from 1 s after the shot on (the shot is 2 s in)
read -r n span <<<"$(ffprobe -v error -select_streams v:0 -show_entries frame=pts_time -of csv=p=0 "$WORK/anim.mkv" 2>/dev/null |
    awk -F, 'NR == 1 { first = $1 } { if ($1 - first > 3) late++; last = $1 } END { printf "%d %.1f\n", late, last - first }')"
[ "${n:-0}" -ge 6 ] && pass "the GIF cover kept playing in the stream after the shot ($n frames later than 1 s after it, ${span} s in all)" ||
    fail "the stream froze after the shot: ${n:-0} frames later than 1 s after it (${span:-?} s in all, $WORK/anim.mkv)"

# The GIF cover's repaints belong to the capture that shows it. Two streams that see the window
# draw no cover; a plain grim shot draws it once, and while it captured, its cover was repainted
# at the GIF's pace (4 frames a second). Once it is over (a capture counts for half a second
# after its frame), nothing on the still screen is repainted: the streams' frames used to keep
# the shot's cover going, and the monitor was repainted at the GIF's pace until a stream stopped.
echo "== animated cover on a still screen, screenshot between two streams that see the window"
sleep 1
"$WORK/wf-share" -f "$WORK/see1.mkv" -c libx264 >/dev/null 2>&1 &
S1=$!
"$WORK/wf-share" -f "$WORK/see2.mkv" -c libx264 >/dev/null 2>&1 &
S2=$!
sleep 2
grim "$WORK/still-plain.png"
sleep 1.5
f0=$(grep -ac "frame: monitor" "$WORK/trace.log")
sleep 3
f1=$(grep -ac "frame: monitor" "$WORK/trace.log")
kill -INT "$S1" "$S2"; wait "$S1" "$S2" 2>/dev/null
sleep 0.6
C=$(mean_rgb "$WORK/still-plain.png" "$BOX")
is_cover "$C" && pass "plain grim gets the GIF cover ($C)" || fail "plain grim sees the window ($C)"
grep -aq "capture client: wf-share" "$WORK/trace.log" || fail "the streams weren't told apart as wf-share ($WORK/trace.log)"
[ $((f1 - f0)) -le 4 ] && pass "the still screen stays still after the shot ($((f1 - f0)) capture frames in 3 s)" ||
    fail "the shot's cover kept the monitor repainting for the streams: $((f1 - f0)) capture frames in 3 s ($WORK/trace.log)"

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed"
exit "$FAILS"
