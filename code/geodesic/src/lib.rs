//! Coding exercises for the unit "Edge-based segmentation: geodesic active contours".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p geodesic`, and run it
//! with `cargo run -p geodesic` (pictures in `out/geodesic/`).
//! `src/solution.rs` is a model solution: `cargo test -p geodesic --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
