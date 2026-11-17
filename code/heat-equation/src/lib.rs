//! Coding exercises for the unit "Smoothing as heat flow".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p heat-equation`, and run it on
//! the images with `cargo run -p heat-equation` (pictures in `out/heat-equation/`).
//! `src/solution.rs` is a model solution: `cargo test -p heat-equation --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
