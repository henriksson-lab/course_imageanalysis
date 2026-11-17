//! Coding exercises for the unit "The multivariate Gaussian and the mixture model".
//!
//! Write your code in `src/gaussian_mixture/exercise.rs`. Check it with `cargo test -p gmm gaussian_mixture::`, and run it on
//! examples with `cargo run -p gmm -- gaussian-mixture` (pictures in `out/gaussian-mixture/`).
//! `src/gaussian_mixture/solution.rs` is a model solution: `cargo test -p gmm --features solution gaussian_mixture::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gmm -- gaussian-mixture`.
pub mod run;
