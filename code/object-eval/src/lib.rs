//! Coding exercises for the unit "Matching objects: IoU, F1 and average precision".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p object-eval`, and run it
//! with `cargo run -p object-eval` (pictures in `out/object-eval/`).
//! `src/solution.rs` is a model solution: `cargo test -p object-eval --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
