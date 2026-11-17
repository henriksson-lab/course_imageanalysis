//! Coding exercises for the unit "Optimisers and learning rates".
//!
//! Write your code in `src/optimisers/exercise.rs`. Check it with `cargo test -p training optimisers::`, and run it
//! with `cargo run -p training -- optimisers` (pictures in `out/optimisers/`).
//! `src/optimisers/solution.rs` is a model solution: `cargo test -p training --features solution optimisers::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p training -- optimisers`.
pub mod run;
