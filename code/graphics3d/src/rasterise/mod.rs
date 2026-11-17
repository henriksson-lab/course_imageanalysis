//! Coding exercises for the unit "From triangles to shaded pixels".
//!
//! Write your code in `src/rasterise/exercise.rs`. Check it with `cargo test -p graphics3d rasterise::`, and run it on
//! a sphere and a textured floor with `cargo run -p graphics3d -- rasterise` (pictures in `out/rasterise/`).
//! `src/rasterise/solution.rs` is a model solution: `cargo test -p graphics3d --features solution rasterise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphics3d -- rasterise`.
pub mod run;
