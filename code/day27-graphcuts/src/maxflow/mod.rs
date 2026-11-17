//! Coding exercises for the unit "Graphs, cuts and maximum flow".
//!
//! Write your code in `src/maxflow/exercise.rs`. Check it with `cargo test -p day27 maxflow::`, and run the
//! examples with `cargo run -p day27 -- maxflow` (pictures in `out/maxflow/`).
//! `src/maxflow/solution.rs` is a model solution: `cargo test -p day27 --features solution maxflow::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day27 -- maxflow`.
pub mod run;
