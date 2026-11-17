//! Coding exercises for the unit "Boosted detectors & the cascade".
//!
//! Write your code in `src/viola_jones/exercise.rs`. Check it with `cargo test -p day22 viola_jones::`, and train a detector
//! for synthetic nuclei with `cargo run -p day22 -- viola-jones` (pictures in `out/viola-jones/`).
//! `src/viola_jones/solution.rs` is a model solution: `cargo test -p day22 --features solution viola_jones::`.

pub mod data;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day22 -- viola-jones`.
pub mod run;
