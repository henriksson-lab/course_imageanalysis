//! Coding exercises for the unit "Convolutional networks".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p cnn`, and run it
//! with `cargo run -p cnn` (pictures in `out/cnn/`).
//! `src/solution.rs` is a model solution: `cargo test -p cnn --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
