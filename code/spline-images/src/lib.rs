//! Coding exercises for the unit "Splines on images: interpolation and free-form deformation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p spline-images`, and run
//! it with `cargo run -p spline-images` (pictures in `out/spline-images/`).
//! `src/solution.rs` is a model solution: `cargo test -p spline-images --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
