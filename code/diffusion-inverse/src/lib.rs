//! Coding exercises for the unit "Diffusion priors for inverse problems, and where detail comes from".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p diffusion-inverse`, and run it
//! with `cargo run -p diffusion-inverse` (pictures in `out/diffusion-inverse/`).
//! `src/solution.rs` is a model solution: `cargo test -p diffusion-inverse --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
