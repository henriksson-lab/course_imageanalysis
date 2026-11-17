//! Coding exercises for the unit "When one threshold is not enough".
//!
//! Write your code in `src/local_threshold/exercise.rs`. Check it with `cargo test -p day12 local_threshold::`, and run it on
//! synthetic images with `cargo run -p day12 -- local-threshold` (pictures in `out/local-threshold/`).
//! `src/local_threshold/solution.rs` is a model solution: `cargo test -p day12 --features solution local_threshold::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day12 -- local-threshold`.
pub mod run;
