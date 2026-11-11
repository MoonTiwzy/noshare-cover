//! `hyprctl clients` and `hyprctl activewindow` with what the plugin does to each window:
//!
//! ```text
//!     noScreenShare: 1
//!     noScreenShareCover: /home/me/.config/hypr/NoCover/stars.mp4
//! ```
//!
//! `noScreenShareCover` is the file that plays, or `transparent`, `blur 24`, `pixelate 20`.
//! Empty on a hidden window: no cover is set and there is no default file, so it gets the
//! black box.
//!
//! Hyprland can't be asked to add fields, so they are written into its finished answer
//! (the shim wraps the two commands). The answer is taken by its shape, a block per window
//! that starts with the window's address; anything that doesn't look like that is left as
//! it is.

use std::collections::HashMap;
use std::ffi::c_char;

use crate::ffi::{guard, opt_str, state};

pub struct Window {
    pub hidden: bool,
    pub cover: String,
}

const HIDDEN: &str = "noScreenShare";
const COVER: &str = "noScreenShareCover";

/// `text` is bytes, not a string: a window title is whatever the client sent.
pub fn annotate(text: &[u8], json: bool, windows: &HashMap<u64, Window>) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + windows.len() * 96);
    // the window whose block the lines belong to
    let mut current: Option<&Window> = None;
    // plain: where its last field ended, the fields go there (before the blank line)
    let mut end = 0;
    for line in text.split_inclusive(|b| *b == b'\n') {
        let body = line.strip_suffix(b"\n").unwrap_or(line);
        if json {
            // strings are escaped, so a line is a field; only a window's own brace isn't indented
            if let Some(found) = json_address(body).and_then(|a| windows.get(&a)) {
                current = Some(found);
            } else if body.starts_with(b"}")
                && let Some(w) = current.take()
                && out.ends_with(b"\n")
            {
                out.pop();
                let cover = escape_json(&w.cover);
                let fields = format!(
                    ",\n    \"{HIDDEN}\": {},\n    \"{COVER}\": \"{cover}\"\n",
                    w.hidden
                );
                out.extend_from_slice(fields.as_bytes());
            }
            out.extend_from_slice(line);
            continue;
        }
        if let Some(address) = plain_address(body) {
            plain_fields(&mut out, end, current.take());
            current = windows.get(&address);
        }
        out.extend_from_slice(line);
        if !body.is_empty() {
            end = out.len();
        }
    }
    if !json {
        plain_fields(&mut out, end, current);
    }
    out
}

fn plain_fields(out: &mut Vec<u8>, at: usize, window: Option<&Window>) {
    let Some(w) = window else { return };
    let fields = format!(
        "\t{HIDDEN}: {}\n\t{COVER}: {}\n",
        u8::from(w.hidden),
        w.cover
    );
    out.splice(at..at, fields.into_bytes());
}

/// `Window 55e45b44c7f0 -> title:`
fn plain_address(line: &[u8]) -> Option<u64> {
    let rest = line.strip_prefix(b"Window ")?;
    let len = rest.iter().position(|b| *b == b' ')?;
    rest[len..].starts_with(b" -> ").then_some(())?;
    hex(&rest[..len])
}

/// `    "address": "0x55e45b44c7f0",`
fn json_address(line: &[u8]) -> Option<u64> {
    hex(line
        .strip_prefix(b"    \"address\": \"0x")?
        .strip_suffix(b"\",")?)
}

fn hex(digits: &[u8]) -> Option<u64> {
    u64::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// One window for `nsc_annotate_clients`. `kind` and `amount`: what covers it, as the shim
/// parsed its rules (0 media, 1 transparent, 2 blur, 3 pixelate).
#[repr(C)]
pub struct NscWindowInfo {
    pub address: u64,
    /// The rule's `no_screen_share_cover`, NULL if it has none.
    pub rule_path: *const c_char,
    pub kind: u32,
    pub amount: i32,
    pub hidden: bool,
}

/// Writes `text` with the plugin's fields for every window in it that `windows` has.
/// Returns the length of the result; it is written only if it fits in `cap`. 0: failed,
/// keep `text`.
///
/// # Safety
/// `text` points to `len` bytes, `windows` to `count` elements, `out` to `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsc_annotate_clients(
    text: *const c_char,
    len: usize,
    json: bool,
    windows: *const NscWindowInfo,
    count: usize,
    out: *mut c_char,
    cap: usize,
) -> usize {
    guard(0, || {
        if text.is_null() || (windows.is_null() && count > 0) {
            return 0;
        }
        let text = unsafe { std::slice::from_raw_parts(text.cast::<u8>(), len) };
        let infos = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(windows, count) }
        };
        let default_play = state().as_ref().map(|reg| reg.default_play());
        let windows = infos
            .iter()
            .map(|w| {
                let cover = match (w.hidden, w.kind) {
                    (false, _) => String::new(),
                    (_, 1) => "transparent".to_owned(),
                    (_, 2) => format!("blur {}", w.amount),
                    (_, 3) => format!("pixelate {}", w.amount),
                    _ => default_play
                        .clone()
                        .unwrap_or_else(|| crate::Settings::default().default_play())
                        .with_rule(unsafe { opt_str(w.rule_path) }, None, None)
                        .path
                        .to_string_lossy()
                        .into_owned(),
                };
                let hidden = w.hidden;
                (w.address, Window { hidden, cover })
            })
            .collect();
        let result = annotate(text, json, &windows);
        if !out.is_null() && result.len() <= cap {
            unsafe {
                std::ptr::copy_nonoverlapping(result.as_ptr(), out.cast::<u8>(), result.len())
            };
        }
        result.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn windows() -> HashMap<u64, Window> {
        let w = |hidden, cover: &str| Window {
            hidden,
            cover: cover.to_owned(),
        };
        HashMap::from([(0xa1, w(true, "/c/\"x\".mp4")), (0xb2, w(false, ""))])
    }

    fn run(text: &str, json: bool) -> String {
        String::from_utf8(annotate(text.as_bytes(), json, &windows())).unwrap()
    }

    #[test]
    fn plain_fields_end_each_block() {
        let text =
            "Window a1 -> one:\n\tmapped: 1\n\tstableID: 18\n\nWindow b2 -> two:\n\tmapped: 1\n\n";
        assert_eq!(
            run(text, false),
            "Window a1 -> one:\n\tmapped: 1\n\tstableID: 18\n\tnoScreenShare: 1\n\tnoScreenShareCover: /c/\"x\".mp4\n\n\
             Window b2 -> two:\n\tmapped: 1\n\tnoScreenShare: 0\n\tnoScreenShareCover: \n\n"
        );
    }

    #[test]
    fn plain_keeps_what_it_does_not_know() {
        for text in [
            "no open windows",
            "Invalid",
            "",
            "Window zz -> x:\n\tmapped: 1\n\n",
        ] {
            assert_eq!(run(text, false), text);
        }
        // a window the shim didn't list, between two it did
        let text =
            "Window a1 -> t:\n\tx: 1\n\nWindow c3 -> t:\n\tx: 1\n\nWindow b2 -> t:\n\tx: 1\n\n";
        let out = run(text, false);
        assert_eq!(out.matches(HIDDEN).count(), 4, "{out}"); // both fields have the word
        assert!(
            out.contains("Window c3 -> t:\n\tx: 1\n\nWindow b2"),
            "{out}"
        );
    }

    #[test]
    fn json_fields_close_each_object() {
        let text = "[{\n    \"address\": \"0xa1\",\n    \"workspace\": {\n        \"address\": \"0xb2\",\n        \"id\": 1\n    },\n    \
                    \"title\": \"}\",\n    \"stableId\": \"18\"\n},{\n    \"address\": \"0xb2\",\n    \"stableId\": \"19\"\n}]";
        assert_eq!(
            run(text, true),
            "[{\n    \"address\": \"0xa1\",\n    \"workspace\": {\n        \"address\": \"0xb2\",\n        \"id\": 1\n    },\n    \
             \"title\": \"}\",\n    \"stableId\": \"18\",\n    \"noScreenShare\": true,\n    \"noScreenShareCover\": \"/c/\\\"x\\\".mp4\"\n},{\n    \
             \"address\": \"0xb2\",\n    \"stableId\": \"19\",\n    \"noScreenShare\": false,\n    \"noScreenShareCover\": \"\"\n}]"
        );
    }

    #[test]
    fn json_keeps_what_it_does_not_know() {
        for text in [
            "{}",
            "[]",
            "[{\n    \"address\": \"0xc3\",\n    \"x\": 1\n}]",
        ] {
            assert_eq!(run(text, true), text);
        }
    }

    #[test]
    fn title_bytes_pass_through() {
        let text = b"Window a1 -> \xff\xfe:\n\ttitle: \xff\xfe\n\n";
        let out = annotate(text, false, &windows());
        assert!(
            out.starts_with(b"Window a1 -> \xff\xfe:\n\ttitle: \xff\xfe\n\tnoScreenShare: 1\n")
        );
    }
}
