//! Coding exercises for the unit "Bleed-through and compensation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p crosstalk`, and run it
//! with `cargo run -p crosstalk` (pictures in `out/crosstalk/`).
//! `src/solution.rs` is a model solution: `cargo test -p crosstalk --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
