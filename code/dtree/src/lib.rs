//! Coding exercises for the unit "Decision trees".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p dtree`, and run it with
//! `cargo run -p dtree` (pictures in `out/dtree/`).
//! `src/solution.rs` is a model solution: `cargo test -p dtree --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
