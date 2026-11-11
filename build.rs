// Embeds the VA-API helper (vaapi-helper) into the plugin.
//
// The Makefile builds libnoshare_cover_vaapi.so first, then the core with
// NSC_VAAPI_HELPER=<path>. Without the variable (cargo test, Windows, builds
// without VA-API) the plugin has no VA-API support and says so in the error.
use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nsc_vaapi_embedded)");
    println!("cargo::rerun-if-env-changed=NSC_VAAPI_HELPER");

    let Some(src) = std::env::var_os("NSC_VAAPI_HELPER").map(PathBuf::from) else {
        return;
    };
    println!("cargo::rerun-if-changed={}", src.display());
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("vaapi-helper.so");
    std::fs::copy(&src, &out).unwrap_or_else(|e| panic!("NSC_VAAPI_HELPER={}: {e}", src.display()));
    println!("cargo::rustc-cfg=nsc_vaapi_embedded");
}
