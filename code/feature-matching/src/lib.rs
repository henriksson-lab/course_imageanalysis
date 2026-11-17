//! Coding exercises for the unit "Matching descriptors".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p feature-matching`, and run it on
//! the micrograph with `cargo run -p feature-matching` (pictures in `out/feature-matching/`).
//! `src/solution.rs` is a model solution: `cargo test -p feature-matching --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
