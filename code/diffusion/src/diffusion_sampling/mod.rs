//! Coding exercises for the unit "Samplers, steps, and judging the samples".
//!
//! Write your code in `src/diffusion_sampling/exercise.rs`. Check it with `cargo test -p diffusion diffusion_sampling::`, and run it
//! with `cargo run -p diffusion -- diffusion-sampling` (pictures in `out/diffusion-sampling/`).
//! `src/diffusion_sampling/solution.rs` is a model solution: `cargo test -p diffusion --features solution diffusion_sampling::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p diffusion -- diffusion-sampling`.
pub mod run;
