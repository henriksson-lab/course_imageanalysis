//! Coding exercises for the unit "Non-rigid registration".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p nonrigid-reg`, and run it on
//! the micrograph with `cargo run -p nonrigid-reg` (pictures in `out/nonrigid-reg/`).
//! `src/solution.rs` is a model solution: `cargo test -p nonrigid-reg --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
