//! Coding exercises for the unit "Denoising, artefacts, and what you may measure".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p denoise`, and run it on
//! a test image with `cargo run -p denoise` (pictures in `out/denoise/`).
//! `src/solution.rs` is a model solution: `cargo test -p denoise --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
