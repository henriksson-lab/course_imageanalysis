//! Coding exercises for the unit "Convolutional networks".
//!
//! Write your code in `src/cnn/exercise.rs`. Check it with `cargo test -p day36 cnn::`, and run it
//! with `cargo run -p day36 -- cnn` (pictures in `out/cnn/`).
//! `src/cnn/solution.rs` is a model solution: `cargo test -p day36 --features solution cnn::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day36 -- cnn`.
pub mod run;
