//! Coding exercises for the unit "Neighbours, parents and children".
//!
//! Write your code in `src/relations/exercise.rs`. Check it with `cargo test -p measurements relations::`, and run it with
//! `cargo run -p measurements -- relations` (pictures in `out/relations/`).
//! `src/relations/solution.rs` is a model solution: `cargo test -p measurements --features solution relations::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p measurements -- relations`.
pub mod run;
