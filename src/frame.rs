//! A decoded frame: either pixels in memory or a dmabuf straight from the GPU decoder.
//!
//! Pixels are premultiplied BGRA in memory (= `DRM_FORMAT_ARGB8888` on little-endian),
//! which is exactly what Hyprland takes in `createTexture(drmFormat, pixels, …)`
//! without repacking. A dmabuf goes to `createTexture(SDMABUFAttrs)` with no CPU copies.

use std::sync::Arc;

/// fourcc `AR24` = `DRM_FORMAT_ARGB8888`.
pub const DRM_FORMAT_ARGB8888: u32 = u32::from_le_bytes(*b"AR24");

#[derive(Debug, Clone)]
pub struct Frame {
    /// Increments with every new source frame. The shim updates the texture
    /// only when this number changes.
    pub generation: u64,
    pub data: FrameData,
}

#[derive(Debug, Clone)]
pub enum FrameData {
    Cpu(CpuFrame),
    #[cfg(target_os = "linux")]
    DmaBuf(Arc<DmaBufFrame>),
}

impl FrameData {
    pub fn size(&self) -> (u32, u32) {
        match self {
            Self::Cpu(f) => (f.width, f.height),
            #[cfg(target_os = "linux")]
            Self::DmaBuf(f) => (f.width, f.height),
        }
    }
}

/// A frame in memory. `Arc` lets us hand the pointer to C++ without a copy and
/// without the decoder overwriting the buffer while a texture is uploaded from it.
#[derive(Debug, Clone)]
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    /// Bytes per row.
    pub stride: u32,
    pub pixels: Arc<[u8]>,
}

impl CpuFrame {
    /// From RGBA8 (as image decoders return it): premultiply + swizzle to BGRA.
    pub fn from_rgba(width: u32, height: u32, rgba: &[u8]) -> Self {
        debug_assert_eq!(rgba.len(), width as usize * height as usize * 4);
        let mut out = Vec::with_capacity(rgba.len());
        for &[r, g, b, a] in rgba.as_chunks::<4>().0 {
            out.extend_from_slice(&[premul(b, a), premul(g, a), premul(r, a), a]);
        }
        Self {
            width,
            height,
            stride: width * 4,
            pixels: out.into(),
        }
    }

    /// A ready BGRA buffer (e.g. the GIF canvas). No copy.
    pub fn from_bgra(width: u32, height: u32, pixels: Arc<[u8]>) -> Self {
        debug_assert_eq!(pixels.len(), width as usize * height as usize * 4);
        Self {
            width,
            height,
            stride: width * 4,
            pixels,
        }
    }
}

/// `c * a / 255` with rounding, as in the original plugin.
#[inline]
pub fn premul(c: u8, a: u8) -> u8 {
    ((u32::from(c) * u32::from(a) + 127) / 255) as u8
}

/// A frame in video memory, exported by the decoder (`vaExportSurfaceHandle`).
/// The fds are closed on drop; the Hyprland texture keeps its own copy.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct DmaBufFrame {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub modifier: u64,
    pub planes: Vec<DmaBufPlane>,
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct DmaBufPlane {
    pub fd: std::os::fd::OwnedFd,
    pub offset: u32,
    pub stride: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiply_and_swizzle() {
        // semi-transparent red -> premultiplied BGRA
        let f = CpuFrame::from_rgba(1, 1, &[255, 0, 0, 128]);
        assert_eq!(&*f.pixels, &[0, 0, 128, 128]);
        assert_eq!(f.stride, 4);
    }

    #[test]
    fn opaque_is_untouched() {
        let f = CpuFrame::from_rgba(1, 1, &[10, 20, 30, 255]);
        assert_eq!(&*f.pixels, &[30, 20, 10, 255]);
    }

    #[test]
    fn fourcc_matches_drm() {
        assert_eq!(DRM_FORMAT_ARGB8888, 0x3432_5241);
    }
}
