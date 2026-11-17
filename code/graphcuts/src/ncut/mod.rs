//! Coding exercises for the unit "Spectral partitioning and normalized cuts".
//!
//! Write your code in `src/ncut/exercise.rs`. Check it with `cargo test -p graphcuts ncut::`, and run the
//! examples with `cargo run -p graphcuts -- ncut` (pictures in `out/ncut/`).
//! `src/ncut/solution.rs` is a model solution: `cargo test -p graphcuts --features solution ncut::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphcuts -- ncut`.
pub mod run;
