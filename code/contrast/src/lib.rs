//! Coding exercises for the unit "Contrast and intensity transforms".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p contrast`, and run it
//! with `cargo run -p contrast` (pictures in `out/contrast/`).
//! `src/solution.rs` is a model solution: `cargo test -p contrast --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
