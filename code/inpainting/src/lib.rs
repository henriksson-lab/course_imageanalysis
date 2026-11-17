//! Coding exercises for the unit "Filling holes".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p inpainting`, and run it on
//! the images with `cargo run -p inpainting` (pictures in `out/inpainting/`).
//! `src/solution.rs` is a model solution: `cargo test -p inpainting --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
