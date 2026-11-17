//! Coding exercises for the unit "Hierarchical segmentation and merge trees".
//!
//! Write your code in `src/merge_trees/exercise.rs`. Check it with `cargo test -p day29 merge_trees::`, and run it with
//! `cargo run -p day29 -- merge-trees` (results in `out/merge-trees/`). `src/merge_trees/solution.rs` is a model solution:
//! `cargo test -p day29 --features solution merge_trees::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day29 -- merge-trees`.
pub mod run;
