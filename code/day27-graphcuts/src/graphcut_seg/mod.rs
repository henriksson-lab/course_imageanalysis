//! Coding exercises for the unit "Graph-cut segmentation".
//!
//! Write your code in `src/graphcut_seg/exercise.rs`. Check it with `cargo test -p day27 graphcut_seg::`, and run the
//! examples with `cargo run -p day27 -- graphcut-seg` (pictures in `out/graphcut-seg/`).
//! `src/graphcut_seg/solution.rs` is a model solution: `cargo test -p day27 --features solution graphcut_seg::`.

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

/// The program of this unit: `cargo run -p day27 -- graphcut-seg`.
pub mod run;
