//! Coding exercises for the unit "Better bases: Legendre moments and reconstruction".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p orthogonal-moments`, and run it
//! with `cargo run -p orthogonal-moments` (results in `out/orthogonal-moments/`).
//! `src/solution.rs` is a model solution: `cargo test -p orthogonal-moments --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
