//! Coding exercises for the unit "Localised basis functions and the Haar transform".
//!
//! Write your code in `src/haar/exercise.rs`. Check it with `cargo test -p wavelets haar::`, and run it
//! with `cargo run -p wavelets -- haar` (pictures in `out/haar/`).
//! `src/haar/solution.rs` is a model solution: `cargo test -p wavelets --features solution haar::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p wavelets -- haar`.
pub mod run;
