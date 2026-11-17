//! Coding exercises for the unit "Sampling, temperature and the phase transition".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p mcmc`, and run it with
//! `cargo run -p mcmc` (results in `out/mcmc/`). `src/solution.rs` is a model solution:
//! `cargo test -p mcmc --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
