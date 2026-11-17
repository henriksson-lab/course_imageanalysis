//! Coding exercises for the unit "Regularisation: what we add, and how much".
//!
//! Write your code in `src/regularise/exercise.rs`. Check it with `cargo test -p deconvolution regularise::`, and run it on
//! a blurred 1D profile with `cargo run -p deconvolution -- regularise` (pictures in `out/regularise/`).
//! `src/regularise/solution.rs` is a model solution: `cargo test -p deconvolution --features solution regularise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p deconvolution -- regularise`.
pub mod run;
