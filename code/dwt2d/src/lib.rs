//! Coding exercises for the unit "The 2D wavelet transform and sparsity of images".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p dwt2d`, and run it
//! with `cargo run -p dwt2d` (pictures in `out/dwt2d/`).
//! `src/solution.rs` is a model solution: `cargo test -p dwt2d --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
