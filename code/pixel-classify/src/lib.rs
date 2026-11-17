//! Coding exercises for the unit "SVMs for pixel classification".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p pixel-classify`, and run it on
//! the course images with `cargo run -p pixel-classify` (pictures in `out/pixel-classify/`).
//! `src/solution.rs` is a model solution: `cargo test -p pixel-classify --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
