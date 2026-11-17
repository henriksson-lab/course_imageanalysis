//! Coding exercises for the unit "Normalising flows".
//!
//! Write your code in `src/flows/exercise.rs`. Check it with `cargo test -p gans flows::`, and run it
//! with `cargo run -p gans -- flows` (pictures in `out/flows/`).
//! `src/flows/solution.rs` is a model solution: `cargo test -p gans --features solution flows::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gans -- flows`.
pub mod run;
