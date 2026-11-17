//! Coding exercises for the unit "Bayesian image denoising with an Ising prior".
//!
//! Write your code in `src/mrf_denoise/exercise.rs`. Check it with `cargo test -p day28 mrf_denoise::`, and run it with
//! `cargo run -p day28 -- mrf-denoise` (results in `out/mrf-denoise/`). `src/mrf_denoise/solution.rs` is a model solution:
//! `cargo test -p day28 --features solution mrf_denoise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day28 -- mrf-denoise`.
pub mod run;
