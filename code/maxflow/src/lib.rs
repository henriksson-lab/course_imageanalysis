//! Coding exercises for the unit "Graphs, cuts and maximum flow".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p maxflow`, and run the
//! examples with `cargo run -p maxflow` (pictures in `out/maxflow/`).
//! `src/solution.rs` is a model solution: `cargo test -p maxflow --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
