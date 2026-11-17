//! Coding exercises for the unit "Finding a shift by sliding and scoring".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p matching`, and run it on
//! the micrograph with `cargo run -p matching` (pictures in `out/matching/`).
//! `src/solution.rs` is a model solution: `cargo test -p matching --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
