//! Coding exercises for the unit "Hierarchical segmentation and merge trees".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p merge-trees`, and run it with
//! `cargo run -p merge-trees` (results in `out/merge-trees/`). `src/solution.rs` is a model solution:
//! `cargo test -p merge-trees --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
