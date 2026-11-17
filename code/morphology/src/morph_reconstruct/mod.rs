//! Coding exercises for the unit "Reconstruction and h-extrema".
//!
//! Write your code in `src/morph_reconstruct/exercise.rs`. Check it with `cargo test -p morphology morph_reconstruct::`, and run it on
//! synthetic nuclei with `cargo run -p morphology -- morph-reconstruct` (pictures in `out/morph-reconstruct/`).
//! `src/morph_reconstruct/solution.rs` is a model solution: `cargo test -p morphology --features solution morph_reconstruct::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p morphology -- morph-reconstruct`.
pub mod run;
