//! Coding exercises for the unit "Active contours (snakes)".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p snakes`, and run it
//! with `cargo run -p snakes` (pictures in `out/snakes/`).
//! `src/solution.rs` is a model solution: `cargo test -p snakes --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
