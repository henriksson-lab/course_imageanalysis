//! Coding exercises for the unit "The snake as a linear system".
//!
//! Write your code in `src/snake_matrix/exercise.rs`. Check it with `cargo test -p snakes snake_matrix::`, and run it
//! with `cargo run -p snakes -- snake-matrix` (pictures in `out/snake-matrix/`).
//! `src/snake_matrix/solution.rs` is a model solution: `cargo test -p snakes --features solution snake_matrix::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p snakes -- snake-matrix`.
pub mod run;
