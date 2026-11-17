//! Coding exercises for the unit "Domain shift and adaptation".
//!
//! Write your code in `src/domain_shift/exercise.rs`. Check it with `cargo test -p finetuning domain_shift::`, and run it
//! with `cargo run -p finetuning -- domain-shift` (pictures in `out/domain-shift/`).
//! `src/domain_shift/solution.rs` is a model solution: `cargo test -p finetuning --features solution domain_shift::`.

/// Given code: the cells, the two microscopes and the classifier.
pub mod model;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p finetuning -- domain-shift`.
pub mod run;
