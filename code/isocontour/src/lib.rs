//! Coding exercises for the unit "Level sets and marching squares".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p isocontour`, and run it
//! with `cargo run -p isocontour` (pictures in `out/isocontour/`).
//! `src/solution.rs` is a model solution: `cargo test -p isocontour --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
