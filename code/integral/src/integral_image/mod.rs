//! Coding exercises for the unit "Integral images".
//!
//! Write your code in `src/integral_image/exercise.rs`. Check it with `cargo test -p integral integral_image::`, and run it on
//! the micrograph with `cargo run -p integral -- integral-image` (pictures in `out/integral-image/`).
//! `src/integral_image/solution.rs` is a model solution: `cargo test -p integral --features solution integral_image::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p integral -- integral-image`.
pub mod run;
