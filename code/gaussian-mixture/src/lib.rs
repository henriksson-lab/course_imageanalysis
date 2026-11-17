//! Coding exercises for the unit "The multivariate Gaussian and the mixture model".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gaussian-mixture`, and run it on
//! examples with `cargo run -p gaussian-mixture` (pictures in `out/gaussian-mixture/`).
//! `src/solution.rs` is a model solution: `cargo test -p gaussian-mixture --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
