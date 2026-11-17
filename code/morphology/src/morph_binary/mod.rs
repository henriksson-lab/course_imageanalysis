//! Coding exercises for the unit "Erosion and dilation".
//!
//! Write your code in `src/morph_binary/exercise.rs`. Check it with `cargo test -p morphology morph_binary::`, and run it on
//! synthetic nuclei with `cargo run -p morphology -- morph-binary` (pictures in `out/morph-binary/`).
//! `src/morph_binary/solution.rs` is a model solution: `cargo test -p morphology --features solution morph_binary::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p morphology -- morph-binary`.
pub mod run;
