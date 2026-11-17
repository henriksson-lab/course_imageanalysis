//! Coding exercises for the unit "Feature spaces and k-means".
//!
//! Write your code in `src/kmeans/exercise.rs`. Check it with `cargo test -p day35 kmeans::`, and run it on
//! examples with `cargo run -p day35 -- kmeans` (pictures in `out/kmeans/`).
//! `src/kmeans/solution.rs` is a model solution: `cargo test -p day35 --features solution kmeans::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day35 -- kmeans`.
pub mod run;
