//! Coding exercises for the unit "Fitting decays with few photons".
//!
//! Write your code in `src/decayfit/exercise.rs`. Check it with `cargo test -p day46 decayfit::`, and run it
//! with `cargo run -p day46 -- decayfit` (pictures in `out/decayfit/`).
//! `src/decayfit/solution.rs` is a model solution: `cargo test -p day46 --features solution decayfit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day46 -- decayfit`.
pub mod run;
