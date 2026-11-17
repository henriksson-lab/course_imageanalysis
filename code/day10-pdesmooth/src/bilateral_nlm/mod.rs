//! Coding exercises for the unit "Bilateral filter and non-local means".
//!
//! Write your code in `src/bilateral_nlm/exercise.rs`. Check it with `cargo test -p day10 bilateral_nlm::`, and run it on
//! the images with `cargo run -p day10 -- bilateral-nlm` (pictures in `out/bilateral-nlm/`).
//! `src/bilateral_nlm/solution.rs` is a model solution: `cargo test -p day10 --features solution bilateral_nlm::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- bilateral-nlm`.
pub mod run;
