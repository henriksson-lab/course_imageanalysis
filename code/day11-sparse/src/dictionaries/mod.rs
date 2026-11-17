//! Coding exercises for the unit "Sparse coding and dictionary learning".
//!
//! Write your code in `src/dictionaries/exercise.rs`. Check it with `cargo test -p day11 dictionaries::`, and run it on
//! the micrograph with `cargo run -p day11 -- dictionaries` (pictures in `out/dictionaries/`).
//! `src/dictionaries/solution.rs` is a model solution: `cargo test -p day11 --features solution dictionaries::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day11 -- dictionaries`.
pub mod run;
