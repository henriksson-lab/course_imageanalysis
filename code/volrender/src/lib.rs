//! Coding exercises for the unit "Seeing voxels: projections and volume rendering".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p volrender`, and run it on
//! a synthetic head phantom with `cargo run -p volrender` (pictures in `out/volrender/`).
//! `src/solution.rs` is a model solution: `cargo test -p volrender --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
