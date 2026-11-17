//! Coding exercises for the unit "Attention, transformers and pretrained models".
//!
//! Write your code in `src/attention/exercise.rs`. Check it with `cargo test -p day39 attention::`, and run it
//! with `cargo run -p day39 -- attention` (pictures in `out/attention/`).
//! `src/attention/solution.rs` is a model solution: `cargo test -p day39 --features solution attention::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day39 -- attention`.
pub mod run;
