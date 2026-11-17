//! Coding exercises for the unit "Samplers, steps, and judging the samples".
//!
//! Write your code in `src/diffusion_sampling/exercise.rs`. Check it with `cargo test -p day41 diffusion_sampling::`, and run it
//! with `cargo run -p day41 -- diffusion-sampling` (pictures in `out/diffusion-sampling/`).
//! `src/diffusion_sampling/solution.rs` is a model solution: `cargo test -p day41 --features solution diffusion_sampling::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day41 -- diffusion-sampling`.
pub mod run;
