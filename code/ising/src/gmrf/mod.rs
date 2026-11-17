//! Coding exercises for the unit "Continuous MRFs: quadratic smoothing and total variation".
//!
//! Write your code in `src/gmrf/exercise.rs`. Check it with `cargo test -p ising gmrf::`, and run it with
//! `cargo run -p ising -- gmrf` (results in `out/gmrf/`). `src/gmrf/solution.rs` is a model solution:
//! `cargo test -p ising --features solution gmrf::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p ising -- gmrf`.
pub mod run;
