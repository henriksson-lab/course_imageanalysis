//! Coding exercises for the unit "Endmembers: the geometry of mixtures".
//!
//! Write your code in `src/endmembers/exercise.rs`. Check it with `cargo test -p unmixing endmembers::`, and run it
//! with `cargo run -p unmixing -- endmembers` (pictures in `out/endmembers/`).
//! `src/endmembers/solution.rs` is a model solution: `cargo test -p unmixing --features solution endmembers::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p unmixing -- endmembers`.
pub mod run;
