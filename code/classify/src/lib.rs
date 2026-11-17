//! Coding exercises for the unit "Linear classifiers".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p classify`, and run it on
//! examples with `cargo run -p classify` (pictures in `out/classify/`).
//! `src/solution.rs` is a model solution: `cargo test -p classify --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
