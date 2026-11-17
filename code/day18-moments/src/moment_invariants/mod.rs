//! Coding exercises for the unit "Normalisation and moment invariants".
//!
//! Write your code in `src/moment_invariants/exercise.rs`. Check it with `cargo test -p day18 moment_invariants::`, and run it
//! with `cargo run -p day18 -- moment-invariants` (results in `out/moment-invariants/`).
//! `src/moment_invariants/solution.rs` is a model solution: `cargo test -p day18 --features solution moment_invariants::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day18 -- moment-invariants`.
pub mod run;
