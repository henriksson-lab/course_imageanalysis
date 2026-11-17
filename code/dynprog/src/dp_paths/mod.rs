//! Coding exercises for the unit "The DP principle: minimal paths on a grid".
//!
//! Write your code in `src/dp_paths/exercise.rs`. Check it with `cargo test -p dynprog dp_paths::`, and run it with
//! `cargo run -p dynprog -- dp-paths` (pictures in `out/dp-paths/`).
//! `src/dp_paths/solution.rs` is a model solution: `cargo test -p dynprog --features solution dp_paths::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p dynprog -- dp-paths`.
pub mod run;
