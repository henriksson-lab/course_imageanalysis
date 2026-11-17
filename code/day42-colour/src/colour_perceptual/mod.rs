//! Coding exercises for the unit "HSV, CIELAB and colour segmentation".
//!
//! Write your code in `src/colour_perceptual/exercise.rs`. Check it with `cargo test -p day42 colour_perceptual::`, and run it
//! with `cargo run -p day42 -- colour-perceptual` (pictures in `out/colour-perceptual/`).
//! `src/colour_perceptual/solution.rs` is a model solution: `cargo test -p day42 --features solution colour_perceptual::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day42 -- colour-perceptual`.
pub mod run;
