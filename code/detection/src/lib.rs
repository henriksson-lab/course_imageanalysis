//! Coding exercises for the unit "Finding objects as boxes".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p detection`, and run it
//! with `cargo run -p detection` (pictures in `out/detection/`).
//! `src/solution.rs` is a model solution: `cargo test -p detection --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
