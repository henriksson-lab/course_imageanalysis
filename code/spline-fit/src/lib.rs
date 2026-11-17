//! Coding exercises for the unit "Interpolating, approximating and smoothing splines".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p spline-fit`, and run it
//! with `cargo run -p spline-fit` (pictures in `out/spline-fit/`).
//! `src/solution.rs` is a model solution: `cargo test -p spline-fit --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
