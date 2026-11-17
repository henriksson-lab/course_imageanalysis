//! Coding exercises for the unit "Solving with ℓ1: thresholding iterations and pursuit".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ista`, and run it
//! on a blurred spike signal with `cargo run -p ista` (pictures in `out/ista/`).
//! `src/solution.rs` is a model solution: `cargo test -p ista --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
