//! Coding exercises for the unit "Sparsity and the ℓ1 penalty".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p sparsity`, and run it on
//! the micrograph with `cargo run -p sparsity` (pictures in `out/sparsity/`).
//! `src/solution.rs` is a model solution: `cargo test -p sparsity --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
