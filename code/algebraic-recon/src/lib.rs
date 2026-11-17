//! Coding exercises for the unit "Reconstruction as a linear system".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p algebraic-recon`, and run it on
//! test images with `cargo run -p algebraic-recon` (pictures in `out/algebraic-recon/`).
//! `src/solution.rs` is a model solution: `cargo test -p algebraic-recon --features solution`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
