//! Coding exercises for the unit "Bayesian image denoising with an Ising prior".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p mrf-denoise`, and run it with
//! `cargo run -p mrf-denoise` (results in `out/mrf-denoise/`). `src/solution.rs` is a model solution:
//! `cargo test -p mrf-denoise --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
