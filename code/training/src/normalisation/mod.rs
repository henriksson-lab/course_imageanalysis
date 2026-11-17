//! Coding exercises for the unit "Initialisation, normalisation and residual connections".
//!
//! Write your code in `src/normalisation/exercise.rs`. Check it with `cargo test -p training normalisation::`, and run it
//! with `cargo run -p training -- normalisation` (pictures in `out/normalisation/`).
//! `src/normalisation/solution.rs` is a model solution: `cargo test -p training --features solution normalisation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p training -- normalisation`.
pub mod run;

/// A deep network with or without residual connections (provided code, used by the program).
pub mod resnet;
