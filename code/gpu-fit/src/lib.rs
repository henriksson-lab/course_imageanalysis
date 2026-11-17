//! Coding exercises for the unit "What GPUs are good and bad at, and precision".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gpu-fit`, and run it
//! with `cargo run -p gpu-fit` (numbers on the terminal, pictures in `out/gpu-fit/`).
//! `src/solution.rs` is a model solution: `cargo test -p gpu-fit --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
