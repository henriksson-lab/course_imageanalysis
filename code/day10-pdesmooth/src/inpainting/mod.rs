//! Coding exercises for the unit "Filling holes".
//!
//! Write your code in `src/inpainting/exercise.rs`. Check it with `cargo test -p day10 inpainting::`, and run it on
//! the images with `cargo run -p day10 -- inpainting` (pictures in `out/inpainting/`).
//! `src/inpainting/solution.rs` is a model solution: `cargo test -p day10 --features solution inpainting::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day10 -- inpainting`.
pub mod run;
