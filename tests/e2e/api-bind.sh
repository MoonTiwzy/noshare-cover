#!/usr/bin/env bash
# noshare_cover_bind (include/noshare_cover_api.h) finds the loaded plugin by "noshare-cover" in
# its path, and takes it only if it exports the API: another library loaded from a directory
# of that name, before the plugin, is passed over. No compositor needed: a stand-in plugin
# exports the API, and the header is built as C and as C++.
#
#   tests/e2e/api-bind.sh
#
# Requires: cc, c++. Exit code 0 means all passed.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/nsc-api.XXXXXX")
FAILS=0
pass() { printf '  \e[32mok\e[0m   %s\n' "$*"; }
fail() { printf '  \e[31mFAIL\e[0m %s\n' "$*"; FAILS=$((FAILS + 1)); }

mkdir -p "$WORK/noshare-cover-decoy" "$WORK/noshare-cover"
# another library in a directory called noshare-cover-something: no API
cat > "$WORK/decoy.c" <<'C'
int decoy_function(void) { return 1; }
C
# the plugin's API, as main.cpp exports it
cat > "$WORK/plugin.c" <<'C'
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
uint32_t noshare_cover_api_version(void) { return 2; }
uint64_t noshare_cover_register_client(const char* name) { (void)name; return 42; }
void noshare_cover_unregister_client(uint64_t client) { (void)client; }
bool noshare_cover_set_rects(uint64_t client, int monitor, const void* rects, size_t count) { (void)client; (void)monitor; (void)rects; (void)count; return true; }
bool noshare_cover_clear_client_rects(uint64_t client) { (void)client; return true; }
void noshare_cover_clear_extra_rects(void) {}
void noshare_cover_add_extra_rect(int monitor, double x, double y, double w, double h, double r) { (void)monitor; (void)x; (void)y; (void)w; (void)h; (void)r; }
C
# loads $1... (RTLD_LOCAL, like Hyprland loads plugins), binds, prints "<result> <version> <client>"
cat > "$WORK/main.c" <<'C'
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* C only gets dl_iterate_phdr with it, before any system header */
#endif
#include "noshare_cover_api.h"
#include <stdio.h>
int main(int argc, char** argv) {
    noshare_cover_api api;
    int               i, rc;
    for (i = 1; i < argc; ++i)
        if (!dlopen(argv[i], RTLD_NOW | RTLD_LOCAL)) {
            fprintf(stderr, "%s\n", dlerror());
            return 9;
        }
    rc = noshare_cover_bind(&api);
    printf("%d %u %llu\n", rc, (unsigned)api.version, api.register_client ? (unsigned long long)api.register_client("test") : 0ULL);
    noshare_cover_unbind(&api);
    return 0;
}
C
cc -shared -fPIC -o "$WORK/noshare-cover-decoy/libdecoy.so" "$WORK/decoy.c" || { fail "can't build the decoy"; exit 1; }
cc -shared -fPIC -o "$WORK/noshare-cover/libnoshare-cover.so" "$WORK/plugin.c" || { fail "can't build the stand-in plugin"; exit 1; }
cc -std=c99 -Wall -Wextra -Werror -I"$ROOT/include" -o "$WORK/bind-c" "$WORK/main.c" -ldl || fail "the header doesn't build as C"
c++ -x c++ -std=c++17 -Wall -Wextra -Werror -I"$ROOT/include" -o "$WORK/bind-cxx" "$WORK/main.c" -ldl || fail "the header doesn't build as C++"

DECOY=$WORK/noshare-cover-decoy/libdecoy.so
PLUGIN=$WORK/noshare-cover/libnoshare-cover.so
for prog in bind-c bind-cxx; do
    [ -x "$WORK/$prog" ] || continue
    out=$("$WORK/$prog" "$DECOY" "$PLUGIN")
    [ "$out" = "0 2 42" ] && pass "$prog: the plugin is found past a library loaded before it from a noshare-cover directory ($out)" ||
        fail "$prog: decoy then plugin: got \"$out\", want \"0 2 42\""
    out=$("$WORK/$prog" "$PLUGIN" "$DECOY")
    [ "$out" = "0 2 42" ] && pass "$prog: plugin then decoy ($out)" || fail "$prog: plugin then decoy: got \"$out\", want \"0 2 42\""
    out=$("$WORK/$prog" "$DECOY")
    [ "$out" = "1 0 0" ] && pass "$prog: the decoy alone is no plugin ($out)" || fail "$prog: decoy alone: got \"$out\", want \"1 0 0\""
    out=$("$WORK/$prog")
    [ "$out" = "1 0 0" ] && pass "$prog: nothing loaded ($out)" || fail "$prog: nothing loaded: got \"$out\", want \"1 0 0\""
done

rm -rf "$WORK"
echo
[ "$FAILS" -eq 0 ] && echo "all passed" || echo "$FAILS failed"
exit "$FAILS"
