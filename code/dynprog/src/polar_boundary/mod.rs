//! Coding exercises for the unit "Closed boundaries: DP in polar coordinates".
//!
//! Write your code in `src/polar_boundary/exercise.rs`. Check it with `cargo test -p dynprog polar_boundary::`, and run it with
//! `cargo run -p dynprog -- polar-boundary` (pictures in `out/polar-boundary/`).
//! `src/polar_boundary/solution.rs` is a model solution: `cargo test -p dynprog --features solution polar_boundary::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p dynprog -- polar-boundary`.
pub mod run;
