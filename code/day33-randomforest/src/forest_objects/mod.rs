//! Coding exercises for the unit "Forests on feature tables, and feature importance".
//!
//! Write your code in `src/forest_objects/exercise.rs`. Check it with `cargo test -p day33 forest_objects::`, and run it with
//! `cargo run -p day33 -- forest-objects` (prints the results).
//! `src/forest_objects/solution.rs` is a model solution: `cargo test -p day33 --features solution forest_objects::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

// A small random forest (from the forest unit; filled in, in `src/common.rs`).
pub use crate::common::{argmax, Forest, Settings, Tree};

/// The program of this unit: `cargo run -p day33 -- forest-objects`.
pub mod run;
