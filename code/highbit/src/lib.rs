//! Coding exercises for the unit "12- and 16-bit images in practice".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p highbit`, and run it
//! with `cargo run -p highbit` (pictures in `out/highbit/`).
//! `src/solution.rs` is a model solution: `cargo test -p highbit --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
