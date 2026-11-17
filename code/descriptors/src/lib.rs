//! Coding exercises for the unit "Describing a keypoint".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p descriptors`, and run it on
//! the micrograph with `cargo run -p descriptors` (pictures in `out/descriptors/`).
//! `src/solution.rs` is a model solution: `cargo test -p descriptors --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
