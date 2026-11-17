//! Coding exercises for the unit "Quantisation and bit depth".
//!
//! Write your code in `src/quantisation/exercise.rs`. Check it with `cargo test -p day03 quantisation::`, and run it
//! with `cargo run -p day03 -- quantisation` (pictures in `out/quantisation/`).
//! `src/quantisation/solution.rs` is a model solution: `cargo test -p day03 --features solution quantisation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day03 -- quantisation`.
pub mod run;
