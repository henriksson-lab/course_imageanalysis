//! Coding exercises for the unit "Solving with ℓ1: thresholding iterations and pursuit".
//!
//! Write your code in `src/ista/exercise.rs`. Check it with `cargo test -p day11 ista::`, and run it
//! on a blurred spike signal with `cargo run -p day11 -- ista` (pictures in `out/ista/`).
//! `src/ista/solution.rs` is a model solution: `cargo test -p day11 --features solution ista::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day11 -- ista`.
pub mod run;
