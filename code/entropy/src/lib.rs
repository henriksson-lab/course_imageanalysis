//! Coding exercises for the unit "Information, entropy and entropy codes".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p entropy`, and run it on
//! the micrograph with `cargo run -p entropy` (pictures in `out/entropy/`).
//! `src/solution.rs` is a model solution: `cargo test -p entropy --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
