//! Coding exercises for the unit "Boosted detectors & the cascade".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p viola-jones`, and train a detector
//! for synthetic nuclei with `cargo run -p viola-jones` (pictures in `out/viola-jones/`).
//! `src/solution.rs` is a model solution: `cargo test -p viola-jones --features solution`.

pub mod data;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
