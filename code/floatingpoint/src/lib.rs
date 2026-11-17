//! Coding exercises for the unit "Floating point".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p floatingpoint`, and run it
//! with `cargo run -p floatingpoint` (pictures in `out/floatingpoint/`).
//! `src/solution.rs` is a model solution: `cargo test -p floatingpoint --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
