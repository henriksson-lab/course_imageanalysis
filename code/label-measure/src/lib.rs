//! Coding exercises for the unit "Working with label images".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p label-measure`, and run it on
//! synthetic nuclei with `cargo run -p label-measure` (pictures in `out/label-measure/`).
//! `src/solution.rs` is a model solution: `cargo test -p label-measure --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
