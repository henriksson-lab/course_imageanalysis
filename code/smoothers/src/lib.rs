//! Coding exercises for the unit "All smoothers on one image".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p smoothers`, and run it on
//! the images with `cargo run -p smoothers` (pictures in `out/smoothers/`).
//! `src/solution.rs` is a model solution: `cargo test -p smoothers --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
