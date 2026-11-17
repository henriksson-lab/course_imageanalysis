//! Coding exercises for the unit "HSV, CIELAB and colour segmentation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p colour-perceptual`, and run it
//! with `cargo run -p colour-perceptual` (pictures in `out/colour-perceptual/`).
//! `src/solution.rs` is a model solution: `cargo test -p colour-perceptual --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
