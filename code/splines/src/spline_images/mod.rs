//! Coding exercises for the unit "Splines on images: interpolation and free-form deformation".
//!
//! Write your code in `src/spline_images/exercise.rs`. Check it with `cargo test -p splines spline_images::`, and run
//! it with `cargo run -p splines -- spline-images` (pictures in `out/spline-images/`).
//! `src/spline_images/solution.rs` is a model solution: `cargo test -p splines --features solution spline_images::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p splines -- spline-images`.
pub mod run;
