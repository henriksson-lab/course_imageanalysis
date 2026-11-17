//! Coding exercises for the unit "Many simple cores: throughput, warps and divergence".
//!
//! Write your code in `src/gpu_parallel/exercise.rs`. Check it with `cargo test -p gpus gpu_parallel::`, and run
//! it with `cargo run -p gpus -- gpu-parallel` (numbers on the terminal, pictures in
//! `out/gpu-parallel/`). `src/gpu_parallel/solution.rs` is a model solution:
//! `cargo test -p gpus --features solution gpu_parallel::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gpus -- gpu-parallel`.
pub mod run;
