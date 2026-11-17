//! Coding exercises for the unit "Detecting objects in one pass: YOLO".
//!
//! Write your code in `src/yolo/exercise.rs`. Check it with `cargo test -p cellnets yolo::`, and run it
//! with `cargo run -p cellnets -- yolo` (pictures in `out/yolo/`).
//! `src/yolo/solution.rs` is a model solution: `cargo test -p cellnets --features solution yolo::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p cellnets -- yolo`.
pub mod run;
