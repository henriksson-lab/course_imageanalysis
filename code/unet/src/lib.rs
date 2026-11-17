//! Coding exercises for the unit "Segmentation networks".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p unet`, and run it
//! with `cargo run -p unet` (pictures in `out/unet/`).
//! `src/solution.rs` is a model solution: `cargo test -p unet --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
