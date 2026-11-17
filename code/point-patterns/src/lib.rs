//! Coding exercises for the unit "Point patterns and complete spatial randomness".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p point-patterns`, and run it
//! with `cargo run -p point-patterns` (results in `out/point-patterns/`).
//! `src/solution.rs` is a model solution: `cargo test -p point-patterns --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
