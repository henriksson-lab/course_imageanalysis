//! Coding exercises for the unit "B-splines and NURBS".
//!
//! Write your code in `src/bsplines/exercise.rs`. Check it with `cargo test -p day06 bsplines::`, and run it
//! with `cargo run -p day06 -- bsplines` (pictures in `out/bsplines/`).
//! `src/bsplines/solution.rs` is a model solution: `cargo test -p day06 --features solution bsplines::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day06 -- bsplines`.
pub mod run;
