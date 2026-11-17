//! Coding exercises for the unit "Polynomial wiggles, piecewise cubics and Bézier curves".
//!
//! Write your code in `src/bezier/exercise.rs`. Check it with `cargo test -p splines bezier::`, and run it
//! with `cargo run -p splines -- bezier` (pictures in `out/bezier/`).
//! `src/bezier/solution.rs` is a model solution: `cargo test -p splines --features solution bezier::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p splines -- bezier`.
pub mod run;
