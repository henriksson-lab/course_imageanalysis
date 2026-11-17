//! Coding exercises for the unit "Kernel functions and the kernel trick".
//!
//! Write your code in `src/kernel_trick/exercise.rs`. Check it with `cargo test -p day34 kernel_trick::`, and run it on
//! examples with `cargo run -p day34 -- kernel-trick` (pictures in `out/kernel-trick/`).
//! `src/kernel_trick/solution.rs` is a model solution: `cargo test -p day34 --features solution kernel_trick::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day34 -- kernel-trick`.
pub mod run;
