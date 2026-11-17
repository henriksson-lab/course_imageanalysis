//! Coding exercises for the unit "Fitting decays with few photons".
//!
//! Write your code in `src/decayfit/exercise.rs`. Check it with `cargo test -p flim decayfit::`, and run it
//! with `cargo run -p flim -- decayfit` (pictures in `out/decayfit/`).
//! `src/decayfit/solution.rs` is a model solution: `cargo test -p flim --features solution decayfit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p flim -- decayfit`.
pub mod run;
