//! Decoders. Software ones are either built in (rav1d) or loaded from the system
//! at runtime (openh264, libvpx); hardware ones (VA-API and NVDEC) also go through
//! dlopen, so the plugin has no hard dependencies and doesn't crash when a library is missing.

#[cfg(feature = "cpu-av1")]
pub mod av1;
#[cfg(feature = "cpu-h264")]
pub mod h264;
#[cfg(feature = "nvdec")]
pub mod nvdec;
#[cfg(all(feature = "vaapi", target_os = "linux"))]
pub mod vaapi;
#[cfg(feature = "cpu-vpx")]
pub mod vpx;
