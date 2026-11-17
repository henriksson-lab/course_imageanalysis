//! Coding exercises for the unit "Networks and classical pipelines compared".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p cellnets-compare`, and run it
//! with `cargo run -p cellnets-compare` (pictures in `out/cellnets-compare/`).
//! `src/solution.rs` is a model solution: `cargo test -p cellnets-compare --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
