//! Coding exercises for the unit "Is the segmentation good? Evaluation against ground truth".
//!
//! Write your code in `src/seg_eval/exercise.rs`. Check it with `cargo test -p day12 seg_eval::`, and run it on
//! synthetic nuclei with `cargo run -p day12 -- seg-eval` (pictures in `out/seg-eval/`).
//! `src/seg_eval/solution.rs` is a model solution: `cargo test -p day12 --features solution seg_eval::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day12 -- seg-eval`.
pub mod run;
