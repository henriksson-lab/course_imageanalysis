//! Coding exercises for the unit "Interpolating, approximating and smoothing splines".
//!
//! Write your code in `src/spline_fit/exercise.rs`. Check it with `cargo test -p splines spline_fit::`, and run it
//! with `cargo run -p splines -- spline-fit` (pictures in `out/spline-fit/`).
//! `src/spline_fit/solution.rs` is a model solution: `cargo test -p splines --features solution spline_fit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p splines -- spline-fit`.
pub mod run;
