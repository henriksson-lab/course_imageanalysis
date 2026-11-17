//! Coding exercises for the unit "From skeleton to graph: pruning and length".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p skel-graph`, and run it with
//! `cargo run -p skel-graph` (pictures in `out/skel-graph/`).
//! `src/solution.rs` is a model solution: `cargo test -p skel-graph --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
