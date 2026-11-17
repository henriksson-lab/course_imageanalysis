//! Coding exercises for the unit "Active contours (snakes)".
//!
//! Write your code in `src/snakes/exercise.rs`. Check it with `cargo test -p snakes snakes::`, and run it
//! with `cargo run -p snakes -- snakes` (pictures in `out/snakes/`).
//! `src/snakes/solution.rs` is a model solution: `cargo test -p snakes --features solution snakes::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p snakes -- snakes`.
pub mod run;
