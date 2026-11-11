#!/usr/bin/env bash
# A video decoded on the GPU the compositor draws with is shown straight from video memory
# (VA-API on Intel and AMD: the GPU's video processor turns each frame into an RGB picture
# of the size it is shown at, and Hyprland imports it as a texture). The capture must look
# like the same video decoded on the CPU: same colours, in a large window and in a small one.
#
#   tests/e2e/video-memory.sh <libnoshare-cover.so>
#
# Requires: Hyprland, grim, wf-recorder, foot, ffmpeg (libx264), jq, imagemagick; a seat. Skipped (exit
# code 0) where frames don't come that way: NVIDIA (NVDEC copies), no VA-API, a compositor
# on another GPU than the decoder. Exit code 0 means all passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-vm.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export NOSHARE_COVER_DEBUG=$WORK/trace.log
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

# colour bars that don't move: every frame of the clip is the same picture
ffmpeg -loglevel error -y -f lavfi -i smptebars=size=640x360:rate=30 -t 3 -c:v libx264 -bf 0 -pix_fmt yuv420p "$WORK/bars.mp4"

hctl() { hyprctl -i 0 "$@"; }
shot() { grim "$WORK/$1.png"; }
win_box() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }
rmse() { magick compare -metric RMSE "$WORK/$1.png" "$WORK/$2.png" null: 2>&1 | sed 's/.*(\(.*\)).*/\1/'; }
below() { awk "BEGIN{exit !($1 < $2)}"; }
# mean colour of a patch in the middle of bar $2 (0..6) of the window, "r g b"
bar() { # $1: shot, $2: bar
    local w h x y
    IFS='x+' read -r w h x y <<<"$BOX"
    magick "$WORK/$1.png" -crop "$((w / 28))x$((h / 8))+$((x + w * (2 * $2 + 1) / 14 - w / 56))+$((y + h / 4))" -resize 1x1\! \
        -format '%[fx:int(255*r)] %[fx:int(255*g)] %[fx:int(255*b)]' info:
}
close() { # two "r g b", every channel within 4
    read -r r1 g1 b1 <<<"$1"; read -r r2 g2 b2 <<<"$2"
    for d in $((r1 - r2)) $((g1 - g2)) $((b1 - b2)); do [ "${d#-}" -le 4 ] || return 1; done
}

write_config() { # $1: backend
    cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$WORK/bars.mp4", backend = "$1" } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    decoration = { rounding = 0, shadow = { enabled = false }, blur = { enabled = false } },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true, disable_watchdog_warning = true },
    debug = { suppress_errors = true },
})
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
LUA
}

write_config cpu
pkill -xu "$(id -u)" Hyprland; sleep 0.5
Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
export WAYLAND_DISPLAY=$(ls -t "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
sleep 2
foot --app-id cover-me sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box cover-me)" ] && break; done
sleep 1.5

compare() { # $1: label; shots cpu-$1 and gpu-$1 exist
    BOX=$(win_box cover-me)
    local d bad=0 i a b
    d=$(rmse "cpu-$1" "gpu-$1")
    # the bars' edges differ a little (the GPU interpolates chroma), the picture doesn't
    below "$d" 0.04 && pass "$1: the capture looks like the CPU's (RMSE $d)" || fail "$1: the capture differs from the CPU's (RMSE $d)"
    for i in 0 1 2 3 4 5 6; do
        a=$(bar "cpu-$1" "$i"); b=$(bar "gpu-$1" "$i")
        close "$a" "$b" || { bad=$((bad + 1)); echo "       bar $i: CPU $a, video memory $b"; }
    done
    [ "$bad" -eq 0 ] && pass "$1: all seven bars have the CPU's colours" || fail "$1: $bad of 7 bars are off"
}

echo "== a large window"
shot cpu-large
write_config auto; hctl reload >/dev/null; sleep 2.5
shot gpu-large
if ! grep -q "from video memory.*imported: ok" "$WORK/trace.log"; then
    echo "  frames don't come in video memory here (see the header): skipped"
    pkill -xu "$(id -u)" Hyprland
    exit 0
fi
pass "frames come in video memory ($(grep -c 'from video memory' "$WORK/trace.log") buffer(s) imported)"
compare large

echo "== a small window: the picture is made smaller on the GPU"
hctl dispatch 'hl.dsp.window.float({ action = "toggle" })' >/dev/null
hctl dispatch 'hl.dsp.window.resize({ x = 300, y = 170 })' >/dev/null
hctl dispatch 'hl.dsp.window.move({ x = 100, y = 100 })' >/dev/null
sleep 0.5
BOX=$(win_box cover-me)
IFS='x+' read -r W H _ _ <<<"$BOX"
if [ "$W" -gt 320 ] || [ "$H" -gt 180 ]; then
    fail "the window didn't get small ($BOX)"
else
    # A cover is decoded for the size it was drawn at lately, so it takes a capture that
    # keeps asking for frames (a stream), and a few seconds for the large window to be forgotten.
    wf-recorder -f "$WORK/stream.mkv" -c libx264 >/dev/null 2>&1 &
    REC=$!
    sleep 6
    shot gpu-small
    grep -q "320x180 from video memory" "$WORK/trace.log" && pass "frames arrive at half the size (320x180 for a $W x $H window)" ||
        fail "frames kept their size for a $W x $H window"
    write_config cpu; hctl reload >/dev/null; sleep 6
    shot cpu-small
    kill -INT "$REC"; wait "$REC" 2>/dev/null
    sleep 0.6
    compare small
fi

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed ($WORK)"
exit "$FAILS"
