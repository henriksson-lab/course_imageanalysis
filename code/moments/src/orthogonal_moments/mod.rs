//! Coding exercises for the unit "Better bases: Legendre moments and reconstruction".
//!
//! Write your code in `src/orthogonal_moments/exercise.rs`. Check it with `cargo test -p moments orthogonal_moments::`, and run it
//! with `cargo run -p moments -- orthogonal-moments` (results in `out/orthogonal-moments/`).
//! `src/orthogonal_moments/solution.rs` is a model solution: `cargo test -p moments --features solution orthogonal_moments::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p moments -- orthogonal-moments`.
pub mod run;
