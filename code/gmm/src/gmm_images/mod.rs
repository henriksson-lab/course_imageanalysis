//! Coding exercises for the unit "Mixtures on images".
//!
//! Write your code in `src/gmm_images/exercise.rs`. Check it with `cargo test -p gmm gmm_images::`, and run it on
//! the course images with `cargo run -p gmm -- gmm-images` (pictures in `out/gmm-images/`).
//! `src/gmm_images/solution.rs` is a model solution: `cargo test -p gmm --features solution gmm_images::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gmm -- gmm-images`.
pub mod run;
