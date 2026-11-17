//! Coding exercises for the unit "Honest validation: leakage, splits and domain shift".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p leakage`, and run it
//! with `cargo run -p leakage` (pictures in `out/leakage/`).
//! `src/solution.rs` is a model solution: `cargo test -p leakage --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
