//! Coding exercises for the unit "Choosing the threshold from the histogram".
//!
//! Write your code in `src/otsu/exercise.rs`. Check it with `cargo test -p day12 otsu::`, and run it on
//! a test image and the micrograph with `cargo run -p day12 -- otsu` (pictures in `out/otsu/`).
//! `src/otsu/solution.rs` is a model solution: `cargo test -p day12 --features solution otsu::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day12 -- otsu`.
pub mod run;
