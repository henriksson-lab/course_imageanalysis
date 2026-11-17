//! Coding exercises for the unit "SVMs for pixel classification".
//!
//! Write your code in `src/pixel_classify/exercise.rs`. Check it with `cargo test -p svm pixel_classify::`, and run it on
//! the course images with `cargo run -p svm -- pixel-classify` (pictures in `out/pixel-classify/`).
//! `src/pixel_classify/solution.rs` is a model solution: `cargo test -p svm --features solution pixel_classify::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p svm -- pixel-classify`.
pub mod run;
