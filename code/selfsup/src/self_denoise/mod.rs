//! Coding exercises for the unit "Self-supervised denoising".
//!
//! Write your code in `src/self_denoise/exercise.rs`. Check it with `cargo test -p selfsup self_denoise::`, and run it
//! with `cargo run -p selfsup -- self-denoise` (pictures in `out/self-denoise/`).
//! `src/self_denoise/solution.rs` is a model solution: `cargo test -p selfsup --features solution self_denoise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p selfsup -- self-denoise`.
pub mod run;
