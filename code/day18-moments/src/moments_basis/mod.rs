//! Coding exercises for the unit "Moments as projections onto monomials".
//!
//! Write your code in `src/moments_basis/exercise.rs`. Check it with `cargo test -p day18 moments_basis::`, and run it
//! with `cargo run -p day18 -- moments-basis` (results in `out/moments-basis/`).
//! `src/moments_basis/solution.rs` is a model solution: `cargo test -p day18 --features solution moments_basis::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day18 -- moments-basis`.
pub mod run;
