//! Vendored from henriksson-lab/YOLOv11-rs, commit 041c9f0277692f32791688b76fb8926cd21697ca
//! (src/model/), unchanged except std::time::Instant -> crate::model::Instant (a no-op on
//! wasm32, where Instant::now() panics). Loss, metrics and training are left out. See README.md.
pub mod anchors;
pub mod attention;
pub mod backbone;
pub mod blocks;
pub mod conv;
pub mod head;
pub mod model;
pub mod neck;
pub mod nms;
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub use shim::Instant;
#[cfg(target_arch = "wasm32")]
mod shim {
    pub struct Instant;
    impl Instant { pub fn now() -> Self { Instant } pub fn elapsed(&self) -> std::time::Duration { std::time::Duration::ZERO } }
}
