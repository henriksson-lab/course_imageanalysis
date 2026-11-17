//! Coding exercises for the unit "Bilateral filter and non-local means".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p bilateral-nlm`, and run it on
//! the images with `cargo run -p bilateral-nlm` (pictures in `out/bilateral-nlm/`).
//! `src/solution.rs` is a model solution: `cargo test -p bilateral-nlm --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
