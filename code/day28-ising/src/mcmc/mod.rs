//! Coding exercises for the unit "Sampling, temperature and the phase transition".
//!
//! Write your code in `src/mcmc/exercise.rs`. Check it with `cargo test -p day28 mcmc::`, and run it with
//! `cargo run -p day28 -- mcmc` (results in `out/mcmc/`). `src/mcmc/solution.rs` is a model solution:
//! `cargo test -p day28 --features solution mcmc::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day28 -- mcmc`.
pub mod run;
