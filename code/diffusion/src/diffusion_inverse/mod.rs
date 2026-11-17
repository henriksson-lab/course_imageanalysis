//! Coding exercises for the unit "Diffusion priors for inverse problems, and where detail comes from".
//!
//! Write your code in `src/diffusion_inverse/exercise.rs`. Check it with `cargo test -p diffusion diffusion_inverse::`, and run it
//! with `cargo run -p diffusion -- diffusion-inverse` (pictures in `out/diffusion-inverse/`).
//! `src/diffusion_inverse/solution.rs` is a model solution: `cargo test -p diffusion --features solution diffusion_inverse::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p diffusion -- diffusion-inverse`.
pub mod run;
