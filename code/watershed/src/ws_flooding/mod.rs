//! Coding exercises for the unit "The flooding algorithm on a grid".
//!
//! Write your code in `src/ws_flooding/exercise.rs`. Check it with `cargo test -p watershed ws_flooding::`, and run it on
//! the exercise grid and a large landscape with `cargo run -p watershed -- ws-flooding` (pictures in `out/ws-flooding/`).
//! `src/ws_flooding/solution.rs` is a model solution: `cargo test -p watershed --features solution ws_flooding::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p watershed -- ws-flooding`.
pub mod run;
