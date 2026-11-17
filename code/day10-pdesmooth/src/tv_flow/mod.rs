//! Coding exercises for the unit "Total variation flow".
//!
//! Write your code in `src/tv_flow/exercise.rs`. Check it with `cargo test -p day10 tv_flow::`, and run it on
//! the images with `cargo run -p day10 -- tv-flow` (pictures in `out/tv-flow/`).
//! `src/tv_flow/solution.rs` is a model solution: `cargo test -p day10 --features solution tv_flow::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- tv-flow`.
pub mod run;
