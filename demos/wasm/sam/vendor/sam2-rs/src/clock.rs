//! wasm patch (demos/wasm/sam/vendor/README.md): `std::time::Instant::now()` panics on
//! wasm32-unknown-unknown, and the decoders call it unconditionally (for optional profiling).
//! Natively this is std's Instant; on wasm32 a stand-in whose elapsed time is always zero.
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant
    }
    pub fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}
