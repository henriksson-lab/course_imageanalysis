//! Coding exercises for the unit "When one threshold is not enough".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p local-threshold`, and run it on
//! synthetic images with `cargo run -p local-threshold` (pictures in `out/local-threshold/`).
//! `src/solution.rs` is a model solution: `cargo test -p local-threshold --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
