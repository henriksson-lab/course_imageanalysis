//! Coding exercises for the unit "Bleed-through and compensation".
//!
//! Write your code in `src/crosstalk/exercise.rs`. Check it with `cargo test -p unmixing crosstalk::`, and run it
//! with `cargo run -p unmixing -- crosstalk` (pictures in `out/crosstalk/`).
//! `src/crosstalk/solution.rs` is a model solution: `cargo test -p unmixing --features solution crosstalk::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p unmixing -- crosstalk`.
pub mod run;
