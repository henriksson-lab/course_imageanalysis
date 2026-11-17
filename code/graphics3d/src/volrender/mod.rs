//! Coding exercises for the unit "Seeing voxels: projections and volume rendering".
//!
//! Write your code in `src/volrender/exercise.rs`. Check it with `cargo test -p graphics3d volrender::`, and run it on
//! a synthetic head phantom with `cargo run -p graphics3d -- volrender` (pictures in `out/volrender/`).
//! `src/volrender/solution.rs` is a model solution: `cargo test -p graphics3d --features solution volrender::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphics3d -- volrender`.
pub mod run;
