//! Coding exercises for the unit "From a peak to a mosaic".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p stitching`, and run it on
//! the micrograph with `cargo run -p stitching` (pictures in `out/stitching/`).
//! `src/solution.rs` is a model solution: `cargo test -p stitching --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
