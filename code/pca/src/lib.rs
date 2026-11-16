//! Coding exercises for the unit "Principal component analysis".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p pca`, and run it on
//! the micrograph with `cargo run -p pca` (pictures in `out/pca/`).
//! `src/solution.rs` is a model solution: `cargo test -p pca --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
