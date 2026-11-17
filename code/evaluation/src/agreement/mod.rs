//! Coding exercises for the unit "Boundaries, counts and the uncertain ground truth".
//!
//! Write your code in `src/agreement/exercise.rs`. Check it with `cargo test -p evaluation agreement::`, and run it
//! with `cargo run -p evaluation -- agreement` (pictures in `out/agreement/`).
//! `src/agreement/solution.rs` is a model solution: `cargo test -p evaluation --features solution agreement::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p evaluation -- agreement`.
pub mod run;
