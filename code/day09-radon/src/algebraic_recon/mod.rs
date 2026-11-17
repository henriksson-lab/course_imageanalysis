//! Coding exercises for the unit "Reconstruction as a linear system".
//!
//! Write your code in `src/algebraic_recon/exercise.rs`. Check it with `cargo test -p day09 algebraic_recon::`, and run it on
//! test images with `cargo run -p day09 -- algebraic-recon` (pictures in `out/algebraic-recon/`).
//! `src/algebraic_recon/solution.rs` is a model solution: `cargo test -p day09 --features solution algebraic_recon::`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day09 -- algebraic-recon`.
pub mod run;
