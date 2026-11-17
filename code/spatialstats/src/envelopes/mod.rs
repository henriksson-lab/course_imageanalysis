//! Coding exercises for the unit "Is it more than chance? Simulation envelopes".
//!
//! Write your code in `src/envelopes/exercise.rs`. Check it with `cargo test -p spatialstats envelopes::`, and run it
//! with `cargo run -p spatialstats -- envelopes` (results in `out/envelopes/`).
//! `src/envelopes/solution.rs` is a model solution: `cargo test -p spatialstats --features solution envelopes::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p spatialstats -- envelopes`.
pub mod run;
