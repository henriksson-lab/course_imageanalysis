//! Coding exercises for the unit "Finding a shift by sliding and scoring".
//!
//! Write your code in `src/matching/exercise.rs`. Check it with `cargo test -p day23 matching::`, and run it on
//! the micrograph with `cargo run -p day23 -- matching` (pictures in `out/matching/`).
//! `src/matching/solution.rs` is a model solution: `cargo test -p day23 --features solution matching::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day23 -- matching`.
pub mod run;
