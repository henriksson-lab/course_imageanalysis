//! Coding exercises for the unit "Is the segmentation good? Evaluation against ground truth".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p seg-eval`, and run it on
//! synthetic nuclei with `cargo run -p seg-eval` (pictures in `out/seg-eval/`).
//! `src/solution.rs` is a model solution: `cargo test -p seg-eval --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
