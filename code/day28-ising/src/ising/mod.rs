//! Coding exercises for the unit "Energy and probability: the Ising model".
//!
//! Write your code in `src/ising/exercise.rs`. Check it with `cargo test -p day28 ising::`, and run it with
//! `cargo run -p day28 -- ising` (results in `out/ising/`). `src/ising/solution.rs` is a model solution:
//! `cargo test -p day28 --features solution ising::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day28 -- ising`.
pub mod run;
