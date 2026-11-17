//! Coding exercises for the unit "Pretext tasks and masked image modelling".
//!
//! Write your code in `src/pretext/exercise.rs`. Check it with `cargo test -p selfsup pretext::`, and run it
//! with `cargo run -p selfsup -- pretext` (pictures in `out/pretext/`).
//! `src/pretext/solution.rs` is a model solution: `cargo test -p selfsup --features solution pretext::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p selfsup -- pretext`.
pub mod run;
