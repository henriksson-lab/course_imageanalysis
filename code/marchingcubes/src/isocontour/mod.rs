//! Coding exercises for the unit "Level sets and marching squares".
//!
//! Write your code in `src/isocontour/exercise.rs`. Check it with `cargo test -p marchingcubes isocontour::`, and run it
//! with `cargo run -p marchingcubes -- isocontour` (pictures in `out/isocontour/`).
//! `src/isocontour/solution.rs` is a model solution: `cargo test -p marchingcubes --features solution isocontour::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p marchingcubes -- isocontour`.
pub mod run;
