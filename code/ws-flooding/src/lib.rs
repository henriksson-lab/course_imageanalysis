//! Coding exercises for the unit "The flooding algorithm on a grid".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ws-flooding`, and run it on
//! the exercise grid and a large landscape with `cargo run -p ws-flooding` (pictures in `out/ws-flooding/`).
//! `src/solution.rs` is a model solution: `cargo test -p ws-flooding --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
