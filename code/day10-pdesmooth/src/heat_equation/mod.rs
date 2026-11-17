//! Coding exercises for the unit "Smoothing as heat flow".
//!
//! Write your code in `src/heat_equation/exercise.rs`. Check it with `cargo test -p day10 heat_equation::`, and run it on
//! the images with `cargo run -p day10 -- heat-equation` (pictures in `out/heat-equation/`).
//! `src/heat_equation/solution.rs` is a model solution: `cargo test -p day10 --features solution heat_equation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- heat-equation`.
pub mod run;
