//! Coding exercises for the unit "Generative adversarial networks".
//!
//! Write your code in `src/gan/exercise.rs`. Check it with `cargo test -p gans gan::`, and run it
//! with `cargo run -p gans -- gan` (pictures in `out/gan/`).
//! `src/gan/solution.rs` is a model solution: `cargo test -p gans --features solution gan::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gans -- gan`.
pub mod run;
