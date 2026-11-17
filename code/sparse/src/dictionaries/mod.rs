//! Coding exercises for the unit "Sparse coding and dictionary learning".
//!
//! Write your code in `src/dictionaries/exercise.rs`. Check it with `cargo test -p sparse dictionaries::`, and run it on
//! the micrograph with `cargo run -p sparse -- dictionaries` (pictures in `out/dictionaries/`).
//! `src/dictionaries/solution.rs` is a model solution: `cargo test -p sparse --features solution dictionaries::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p sparse -- dictionaries`.
pub mod run;
