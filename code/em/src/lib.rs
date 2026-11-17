//! Coding exercises for the unit "Fitting a mixture: the EM algorithm".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p em`, and run it on
//! examples with `cargo run -p em` (pictures in `out/em/`).
//! `src/solution.rs` is a model solution: `cargo test -p em --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
