//! Coding exercises for the unit "Fitting decays with few photons".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p decayfit`, and run it
//! with `cargo run -p decayfit` (pictures in `out/decayfit/`).
//! `src/solution.rs` is a model solution: `cargo test -p decayfit --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
