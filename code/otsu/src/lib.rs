//! Coding exercises for the unit "Choosing the threshold from the histogram".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p otsu`, and run it on
//! a test image and the micrograph with `cargo run -p otsu` (pictures in `out/otsu/`).
//! `src/solution.rs` is a model solution: `cargo test -p otsu --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
