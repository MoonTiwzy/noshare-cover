#!/usr/bin/env bash
# A cover that plays keeps a still monitor repainting while it is captured, at the cover's
# own pace: a stream of a 10 fps video cover gets about 10 frames a second and a 30 fps one
# about 30 (not the 60 of a fixed timer), and a picture that doesn't change gets none.
#
#   tests/e2e/pump.sh <libnoshare-cover.so>
#
# Requires: Hyprland, foot, wf-recorder, ffmpeg and ffprobe (libx264), jq, imagemagick; a
# seat. Exit code 0 means all passed.
set -uo pipefail

PLUGIN=$(realpath "${1:?path to libnoshare-cover.so}")
WORK=$(mktemp -d /tmp/nsc-pump.XXXXXX)
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
export NOSHARE_COVER_DEBUG=$WORK/trace.log
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

clip() { # $1: fps, $2: file
    ffmpeg -loglevel error -y -f lavfi -i "testsrc2=size=320x180:rate=$1" -t 4 -c:v libx264 -bf 0 -pix_fmt yuv420p "$2"
}
clip 10 "$WORK/slow.mp4"
clip 30 "$WORK/fast.mp4"
magick -size 320x180 gradient:'#ff00ff-#00ffff' "$WORK/still.png"

hctl() { hyprctl -i 0 "$@"; }
win_box() { hctl clients -j | jq -r --arg c "$1" '.[] | select(.class == $c) | "\(.size[0])x\(.size[1])+\(.at[0])+\(.at[1])"' | head -1; }

write_config() { # $1: the cover
    cat > "$WORK/hypr.lua" <<LUA
hl.monitor({ output = "", mode = "preferred", position = "auto", scale = "1" })
hl.plugin.load("$PLUGIN")
hl.config({
    plugin = { no_screen_share_cover = { path_cover = "$1" } },
    general = { gaps_in = 0, gaps_out = 0, border_size = 0, layout = "dwindle" },
    decoration = { rounding = 0, shadow = { enabled = false }, blur = { enabled = false } },
    animations = { enabled = false },
    misc = { disable_hyprland_logo = true, disable_splash_rendering = true, disable_watchdog_warning = true },
    debug = { suppress_errors = true },
})
hl.window_rule({ match = { class = "cover-me" }, no_screen_share = true })
LUA
}

write_config "$WORK/slow.mp4"
pkill -xu "$(id -u)" Hyprland; sleep 0.5
Hyprland --config "$WORK/hypr.lua" ${HYPR_ARGS:-} > "$WORK/hypr.log" 2>&1 &
for _ in $(seq 60); do sleep 0.25; hctl version >/dev/null 2>&1 && break; done
export WAYLAND_DISPLAY=$(ls -t "$XDG_RUNTIME_DIR" | grep -m1 '^wayland-[0-9]*$')
sleep 2
# a window that never repaints by itself: no cursor, no output
foot --app-id cover-me sh -c 'printf "\033[?25l"; sleep 1000' >/dev/null 2>&1 &
for _ in $(seq 40); do sleep 0.25; [ -n "$(win_box cover-me)" ] && break; done
sleep 1.5

# frames per second of a 5 s stream, from the frames' own times
stream_fps() { # $1: name
    wf-recorder -f "$WORK/$1.mkv" -c libx264 >/dev/null 2>&1 &
    local rec=$!
    sleep 5
    kill -INT "$rec"; wait "$rec" 2>/dev/null
    sleep 0.6
    ffprobe -v error -select_streams v:0 -show_entries frame=pts_time -of csv=p=0 "$WORK/$1.mkv" 2>/dev/null |
        awk -F, 'NR == 1 { first = $1 } { last = $1; n++ } END { if (n < 2 || last <= first) print n + 0, 0; else printf "%d %.1f\n", n, (n - 1) / (last - first) }'
}
between() { awk "BEGIN{exit !($1 >= $2 && $1 <= $3)}"; }
frame() { # $1: stream, $2: its frame number -> $WORK/$1-$2.png
    ffmpeg -loglevel error -y -i "$WORK/$1.mkv" -vf "select=eq(n\\,$2)" -frames:v 1 "$WORK/$1-$2.png" 2>/dev/null
}

check_video() { # $1: clip, $2: its fps, $3: lowest fps, $4: highest fps
    write_config "$WORK/$1.mp4"; hctl reload >/dev/null; sleep 2
    local n fps
    read -r n fps <<<"$(stream_fps "$1")"
    between "$fps" "$3" "$4" && pass "$2 fps cover: the stream gets $fps frames a second ($n frames)" ||
        fail "$2 fps cover: the stream gets $fps frames a second ($n frames), expected $3..$4"
    # and they are the video: two of them a second apart differ
    local a=5 b=$((5 + $2)) d
    frame "$1" "$a"; frame "$1" "$b"
    d=$(magick compare -metric RMSE "$WORK/$1-$a.png" "$WORK/$1-$b.png" null: 2>&1 | sed 's/.*(\(.*\)).*/\1/')
    awk "BEGIN{exit !(${d:-0} > 0.005)}" 2>/dev/null && pass "$2 fps cover: it plays in the stream (RMSE between frames $d)" ||
        fail "$2 fps cover: frames a second apart are the same (RMSE ${d:-none})"
}

echo "== a video cover sets the pace"
check_video slow 10 7 16
check_video fast 30 22 40

echo "== a still cover leaves the monitor alone"
write_config "$WORK/still.png"; hctl reload >/dev/null; sleep 2
read -r n fps <<<"$(stream_fps still)"
[ "$n" -le 5 ] && pass "still cover: $n frames in 5 s" || fail "still cover: the monitor kept repainting ($n frames in 5 s)"

pkill -xu "$(id -u)" Hyprland
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed ($WORK)"
exit "$FAILS"
