//! Coding exercises for the unit "Feature spaces and k-means".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p kmeans`, and run it on
//! examples with `cargo run -p kmeans` (pictures in `out/kmeans/`).
//! `src/solution.rs` is a model solution: `cargo test -p kmeans --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
