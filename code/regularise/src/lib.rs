//! Coding exercises for the unit "Regularisation: what we add, and how much".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p regularise`, and run it on
//! a blurred 1D profile with `cargo run -p regularise` (pictures in `out/regularise/`).
//! `src/solution.rs` is a model solution: `cargo test -p regularise --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
