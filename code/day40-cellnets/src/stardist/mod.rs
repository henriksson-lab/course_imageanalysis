//! Coding exercises for the unit "Star-convex polygons".
//!
//! Write your code in `src/stardist/exercise.rs`. Check it with `cargo test -p day40 stardist::`, and run it
//! with `cargo run -p day40 -- stardist` (pictures in `out/stardist/`).
//! `src/stardist/solution.rs` is a model solution: `cargo test -p day40 --features solution stardist::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day40 -- stardist`.
pub mod run;
