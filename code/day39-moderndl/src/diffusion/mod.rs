//! Coding exercises for the unit "Diffusion models and generated detail".
//!
//! Write your code in `src/diffusion/exercise.rs`. Check it with `cargo test -p day39 diffusion::`, and run it
//! with `cargo run -p day39 -- diffusion` (pictures in `out/diffusion/`).
//! `src/diffusion/solution.rs` is a model solution: `cargo test -p day39 --features solution diffusion::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day39 -- diffusion`.
pub mod run;
