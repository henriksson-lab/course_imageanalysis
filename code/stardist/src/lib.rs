//! Coding exercises for the unit "Star-convex polygons".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p stardist`, and run it
//! with `cargo run -p stardist` (pictures in `out/stardist/`).
//! `src/solution.rs` is a model solution: `cargo test -p stardist --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
