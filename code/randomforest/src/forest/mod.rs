//! Coding exercises for the unit "Bagging and random forests".
//!
//! Write your code in `src/forest/exercise.rs`. Check it with `cargo test -p randomforest forest::`, and run it with
//! `cargo run -p randomforest -- forest` (pictures in `out/forest/`).
//! `src/forest/solution.rs` is a model solution: `cargo test -p randomforest --features solution forest::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// The decision trees (from the decision-trees unit; filled in).
mod tree;
pub use tree::{grow_tree, Tree};

/// The program of this unit: `cargo run -p randomforest -- forest`.
pub mod run;
