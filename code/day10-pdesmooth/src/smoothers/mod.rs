//! Coding exercises for the unit "All smoothers on one image".
//!
//! Write your code in `src/smoothers/exercise.rs`. Check it with `cargo test -p day10 smoothers::`, and run it on
//! the images with `cargo run -p day10 -- smoothers` (pictures in `out/smoothers/`).
//! `src/smoothers/solution.rs` is a model solution: `cargo test -p day10 --features solution smoothers::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- smoothers`.
pub mod run;
