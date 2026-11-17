//! Coding exercises for the unit "Region-based segmentation: Chan–Vese".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p chanvese`, and run it
//! with `cargo run -p chanvese` (pictures in `out/chanvese/`).
//! `src/solution.rs` is a model solution: `cargo test -p chanvese --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
