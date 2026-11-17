//! Coding exercises for the unit "Memory is the bottleneck: bandwidth, coalescing, transfers
//! and the roofline".
//!
//! Write your code in `src/gpu_memory/exercise.rs`. Check it with `cargo test -p gpus gpu_memory::`, and run it
//! with `cargo run -p gpus -- gpu-memory` (numbers on the terminal, pictures in `out/gpu-memory/`).
//! `src/gpu_memory/solution.rs` is a model solution: `cargo test -p gpus --features solution gpu_memory::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gpus -- gpu-memory`.
pub mod run;
