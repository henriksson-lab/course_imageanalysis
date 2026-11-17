//! Coding exercises for the unit "Histograms and a global threshold".
//!
//! Write your code in `src/histogram/exercise.rs`. Check it with `cargo test -p thresholding histogram::`, and run it on
//! synthetic blobs and the micrograph with `cargo run -p thresholding -- histogram` (pictures in `out/histogram/`).
//! `src/histogram/solution.rs` is a model solution: `cargo test -p thresholding --features solution histogram::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p thresholding -- histogram`.
pub mod run;
