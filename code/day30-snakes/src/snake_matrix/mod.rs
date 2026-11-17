//! Coding exercises for the unit "The snake as a linear system".
//!
//! Write your code in `src/snake_matrix/exercise.rs`. Check it with `cargo test -p day30 snake_matrix::`, and run it
//! with `cargo run -p day30 -- snake-matrix` (pictures in `out/snake-matrix/`).
//! `src/snake_matrix/solution.rs` is a model solution: `cargo test -p day30 --features solution snake_matrix::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day30 -- snake-matrix`.
pub mod run;
