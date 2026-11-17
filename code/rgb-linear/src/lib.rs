//! Coding exercises for the unit "RGB: gamma, linear maps and white balance".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p rgb-linear`, and run it
//! with `cargo run -p rgb-linear` (pictures in `out/rgb-linear/`).
//! `src/solution.rs` is a model solution: `cargo test -p rgb-linear --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
