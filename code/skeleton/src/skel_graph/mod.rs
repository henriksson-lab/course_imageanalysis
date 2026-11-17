//! Coding exercises for the unit "From skeleton to graph: pruning and length".
//!
//! Write your code in `src/skel_graph/exercise.rs`. Check it with `cargo test -p skeleton skel_graph::`, and run it with
//! `cargo run -p skeleton -- skel-graph` (pictures in `out/skel-graph/`).
//! `src/skel_graph/solution.rs` is a model solution: `cargo test -p skeleton --features solution skel_graph::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p skeleton -- skel-graph`.
pub mod run;
