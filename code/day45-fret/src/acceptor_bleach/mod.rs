//! Coding exercises for the unit "Acceptor photobleaching, and what an efficiency means".
//!
//! Write your code in `src/acceptor_bleach/exercise.rs`. Check it with `cargo test -p day45 acceptor_bleach::`, and run it with
//! `cargo run -p day45 -- acceptor-bleach` (results in `out/acceptor-bleach/`).
//! `src/acceptor_bleach/solution.rs` is a model solution: `cargo test -p day45 --features solution acceptor_bleach::`.
//! `src/sim.rs` makes the test data and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day45 -- acceptor-bleach`.
pub mod run;
