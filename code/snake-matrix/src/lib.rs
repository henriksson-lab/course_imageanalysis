//! Coding exercises for the unit "The snake as a linear system".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p snake-matrix`, and run it
//! with `cargo run -p snake-matrix` (pictures in `out/snake-matrix/`).
//! `src/solution.rs` is a model solution: `cargo test -p snake-matrix --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
