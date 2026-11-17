//! Coding exercises for the unit "Total variation flow".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p tv-flow`, and run it on
//! the images with `cargo run -p tv-flow` (pictures in `out/tv-flow/`).
//! `src/solution.rs` is a model solution: `cargo test -p tv-flow --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
