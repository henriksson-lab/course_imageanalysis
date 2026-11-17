//! Coding exercises for the unit "Shape models: landmarks, Procrustes and PCA".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p shape-models`, and run it
//! with `cargo run -p shape-models` (pictures in `out/shape-models/`).
//! `src/solution.rs` is a model solution: `cargo test -p shape-models --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
