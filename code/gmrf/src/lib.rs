//! Coding exercises for the unit "Continuous MRFs: quadratic smoothing and total variation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gmrf`, and run it with
//! `cargo run -p gmrf` (results in `out/gmrf/`). `src/solution.rs` is a model solution:
//! `cargo test -p gmrf --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
