//! Coding exercises for the unit "Data-parallel patterns: map, stencil, reduction, scan and
//! histogram".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gpu-patterns`, and run
//! it with `cargo run -p gpu-patterns` (numbers on the terminal, pictures in
//! `out/gpu-patterns/`). `src/solution.rs` is a model solution:
//! `cargo test -p gpu-patterns --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
