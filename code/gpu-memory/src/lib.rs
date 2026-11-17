//! Coding exercises for the unit "Memory is the bottleneck: bandwidth, coalescing, transfers
//! and the roofline".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gpu-memory`, and run it
//! with `cargo run -p gpu-memory` (numbers on the terminal, pictures in `out/gpu-memory/`).
//! `src/solution.rs` is a model solution: `cargo test -p gpu-memory --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
