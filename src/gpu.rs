//! GPU selection for hardware decoding.
//!
//! `gpu_device` in the config is an explicit render node (`/dev/dri/renderD129`).
//! If empty, the GPU the compositor draws with comes first when it is known: frames
//! decoded there can be shown without leaving video memory, and that GPU is awake
//! anyway. Then the other render nodes, lowest number first. No environment variables.

use std::path::{Path, PathBuf};

/// The GPU behind a DRM device number, as its sysfs path: the same for a GPU's card
/// node and its render node.
pub fn device_of_dev(dev: u64) -> Option<PathBuf> {
    if dev == 0 {
        return None;
    }
    let (major, minor) = (libc::major(dev), libc::minor(dev));
    std::fs::canonicalize(format!("/sys/dev/char/{major}:{minor}/device")).ok()
}

/// The GPU behind a DRM node (`/dev/dri/renderD128`), see [`device_of_dev`].
pub fn device_of_node(node: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    device_of_dev(std::fs::metadata(node).ok()?.rdev())
}

/// The render nodes to decode on, in the order to try them: only `explicit` if it is
/// set, otherwise the compositor's GPU (`compositor`, see [`device_of_dev`]) and then
/// the others. Never empty.
pub fn render_node_order(
    explicit: Option<&Path>,
    compositor: Option<&Path>,
) -> Result<Vec<PathBuf>, String> {
    if let Some(p) = explicit {
        return if p.exists() {
            Ok(vec![p.to_path_buf()])
        } else {
            Err(format!("no such device: {}", p.display()))
        };
    }
    let mut nodes = render_nodes(Path::new("/dev/dri"))?;
    if nodes.is_empty() {
        return Err("no render node found".into());
    }
    if let Some(c) = compositor
        && let Some(own) = nodes
            .iter()
            .position(|n| device_of_node(n).as_deref() == Some(c))
    {
        let node = nodes.remove(own);
        nodes.insert(0, node);
    }
    Ok(nodes)
}

/// The render nodes in `dir`, lowest number first.
fn render_nodes(dir: &Path) -> Result<Vec<PathBuf>, String> {
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
    Ok(nodes.into_iter().map(|(_, p)| p).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_missing_device_is_error() {
        assert!(render_node_order(Some(Path::new("/nonexistent/renderD128")), None).is_err());
    }

    #[test]
    fn picks_lowest_render_node() {
        let dir = std::env::temp_dir().join(format!("nsc-dri-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["renderD129", "renderD128", "card0"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        assert_eq!(
            render_nodes(&dir).unwrap(),
            [dir.join("renderD128"), dir.join("renderD129")]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_gpu_is_the_same_through_any_of_its_nodes() {
        assert_eq!(device_of_dev(0), None);
        assert_eq!(device_of_node(Path::new("/nonexistent/renderD128")), None);
        // on a machine with a GPU: its render node leads to a device, and with that device
        // as the compositor's, its node is tried first and the others still follow
        let Ok(nodes) = render_nodes(Path::new("/dev/dri")) else {
            return;
        };
        for node in &nodes {
            let Some(dev) = device_of_node(node) else {
                continue;
            };
            let order = render_node_order(None, Some(&dev)).unwrap();
            assert_eq!(device_of_node(&order[0]).as_deref(), Some(dev.as_path()));
            assert_eq!(order.len(), nodes.len());
            // an explicit node is the only one
            assert_eq!(
                render_node_order(Some(node), Some(&dev)).unwrap(),
                std::slice::from_ref(node)
            );
        }
        if let Ok(order) = render_node_order(None, None) {
            assert_eq!(order, nodes, "nothing known: lowest number first");
        }
    }
}
