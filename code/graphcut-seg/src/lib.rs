//! Coding exercises for the unit "Graph-cut segmentation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p graphcut-seg`, and run the
//! examples with `cargo run -p graphcut-seg` (pictures in `out/graphcut-seg/`).
//! `src/solution.rs` is a model solution: `cargo test -p graphcut-seg --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// Provided: maximum flow on large sparse graphs (not an exercise; see the maxflow unit).
mod flow;
pub use flow::FlowGraph;
