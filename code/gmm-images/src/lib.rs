//! Coding exercises for the unit "Mixtures on images".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gmm-images`, and run it on
//! the course images with `cargo run -p gmm-images` (pictures in `out/gmm-images/`).
//! `src/solution.rs` is a model solution: `cargo test -p gmm-images --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
