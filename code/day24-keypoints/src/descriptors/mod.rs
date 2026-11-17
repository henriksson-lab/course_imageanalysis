//! Coding exercises for the unit "Describing a keypoint".
//!
//! Write your code in `src/descriptors/exercise.rs`. Check it with `cargo test -p day24 descriptors::`, and run it on
//! the micrograph with `cargo run -p day24 -- descriptors` (pictures in `out/descriptors/`).
//! `src/descriptors/solution.rs` is a model solution: `cargo test -p day24 --features solution descriptors::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day24 -- descriptors`.
pub mod run;
