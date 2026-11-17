//! Coding exercises for the unit "What GPUs are good and bad at, and precision".
//!
//! Write your code in `src/gpu_fit/exercise.rs`. Check it with `cargo test -p day51 gpu_fit::`, and run it
//! with `cargo run -p day51 -- gpu-fit` (numbers on the terminal, pictures in `out/gpu-fit/`).
//! `src/gpu_fit/solution.rs` is a model solution: `cargo test -p day51 --features solution gpu_fit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day51 -- gpu-fit`.
pub mod run;
