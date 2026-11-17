//! Coding exercises for the unit "Many simple cores: throughput, warps and divergence".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gpu-parallel`, and run
//! it with `cargo run -p gpu-parallel` (numbers on the terminal, pictures in
//! `out/gpu-parallel/`). `src/solution.rs` is a model solution:
//! `cargo test -p gpu-parallel --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
