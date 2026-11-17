//! Coding exercises for the unit "Data-parallel patterns: map, stencil, reduction, scan and
//! histogram".
//!
//! Write your code in `src/gpu_patterns/exercise.rs`. Check it with `cargo test -p day51 gpu_patterns::`, and run
//! it with `cargo run -p day51 -- gpu-patterns` (numbers on the terminal, pictures in
//! `out/gpu-patterns/`). `src/gpu_patterns/solution.rs` is a model solution:
//! `cargo test -p day51 --features solution gpu_patterns::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day51 -- gpu-patterns`.
pub mod run;
