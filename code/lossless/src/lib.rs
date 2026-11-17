//! Coding exercises for the unit "Prediction, dictionaries and lossless formats".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p lossless`, and run it on
//! the micrograph with `cargo run -p lossless` (pictures in `out/lossless/`).
//! `src/solution.rs` is a model solution: `cargo test -p lossless --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
