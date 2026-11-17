//! Coding exercises for the unit "Kernel functions and the kernel trick".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p kernel-trick`, and run it on
//! examples with `cargo run -p kernel-trick` (pictures in `out/kernel-trick/`).
//! `src/solution.rs` is a model solution: `cargo test -p kernel-trick --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
