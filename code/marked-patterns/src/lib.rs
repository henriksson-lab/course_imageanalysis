//! Coding exercises for the unit "Two cell types and an uneven density".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p marked-patterns`, and run it
//! with `cargo run -p marked-patterns` (results in `out/marked-patterns/`).
//! `src/solution.rs` is a model solution: `cargo test -p marked-patterns --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
