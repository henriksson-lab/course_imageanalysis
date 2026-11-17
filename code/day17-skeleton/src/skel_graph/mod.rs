//! Coding exercises for the unit "From skeleton to graph: pruning and length".
//!
//! Write your code in `src/skel_graph/exercise.rs`. Check it with `cargo test -p day17 skel_graph::`, and run it with
//! `cargo run -p day17 -- skel-graph` (pictures in `out/skel-graph/`).
//! `src/skel_graph/solution.rs` is a model solution: `cargo test -p day17 --features solution skel_graph::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day17 -- skel-graph`.
pub mod run;
