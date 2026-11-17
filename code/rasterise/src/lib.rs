//! Coding exercises for the unit "From triangles to shaded pixels".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p rasterise`, and run it on
//! a sphere and a textured floor with `cargo run -p rasterise` (pictures in `out/rasterise/`).
//! `src/solution.rs` is a model solution: `cargo test -p rasterise --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
