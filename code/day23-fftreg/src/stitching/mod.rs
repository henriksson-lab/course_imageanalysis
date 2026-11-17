//! Coding exercises for the unit "From a peak to a mosaic".
//!
//! Write your code in `src/stitching/exercise.rs`. Check it with `cargo test -p day23 stitching::`, and run it on
//! the micrograph with `cargo run -p day23 -- stitching` (pictures in `out/stitching/`).
//! `src/stitching/solution.rs` is a model solution: `cargo test -p day23 --features solution stitching::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day23 -- stitching`.
pub mod run;
