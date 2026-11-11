//! GPU selection for hardware decoding.
//!
//! `gpu_device` in the config is an explicit render node (`/dev/dri/renderD129`).
//! If empty, the lowest-numbered render node is used. No environment variables.

use std::path::{Path, PathBuf};

pub fn pick_render_node(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return if p.exists() {
            Ok(p.to_path_buf())
        } else {
            Err(format!("no such device: {}", p.display()))
        };
    }
    first_render_node(Path::new("/dev/dri"))
}

fn first_render_node(dir: &Path) -> Result<PathBuf, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|_| format!("no {} (GPU unavailable)", dir.display()))?;
    let mut nodes: Vec<(u32, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let n = name.strip_prefix("renderD")?.parse().ok()?;
            Some((n, e.path()))
        })
        .collect();
    nodes.sort_by_key(|(n, _)| *n);
    nodes
        .into_iter()
        .next()
        .map(|(_, p)| p)
        .ok_or_else(|| "no render node found".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_missing_device_is_error() {
        assert!(pick_render_node(Some(Path::new("/nonexistent/renderD128"))).is_err());
    }

    #[test]
    fn picks_lowest_render_node() {
        let dir = std::env::temp_dir().join(format!("nsc-dri-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["renderD129", "renderD128", "card0"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        assert_eq!(first_render_node(&dir).unwrap(), dir.join("renderD128"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
