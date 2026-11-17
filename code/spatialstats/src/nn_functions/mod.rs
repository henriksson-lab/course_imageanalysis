//! Coding exercises for the unit "Nearest neighbours and empty space: G, F and J".
//!
//! Write your code in `src/nn_functions/exercise.rs`. Check it with `cargo test -p spatialstats nn_functions::`, and run it
//! with `cargo run -p spatialstats -- nn-functions` (results in `out/nn-functions/`).
//! `src/nn_functions/solution.rs` is a model solution: `cargo test -p spatialstats --features solution nn_functions::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p spatialstats -- nn-functions`.
pub mod run;
