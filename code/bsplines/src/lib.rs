//! Coding exercises for the unit "B-splines and NURBS".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p bsplines`, and run it
//! with `cargo run -p bsplines` (pictures in `out/bsplines/`).
//! `src/solution.rs` is a model solution: `cargo test -p bsplines --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
