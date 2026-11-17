//! Coding exercises for the unit "Forests on feature tables, and feature importance".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p forest-objects`, and run it with
//! `cargo run -p forest-objects` (prints the results).
//! `src/solution.rs` is a model solution: `cargo test -p forest-objects --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// A small random forest (from the forest unit; filled in).
pub mod forest;
pub use forest::{argmax, Forest, Settings, Tree};
