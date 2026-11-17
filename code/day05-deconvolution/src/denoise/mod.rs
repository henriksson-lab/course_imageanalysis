//! Coding exercises for the unit "Denoising, artefacts, and what you may measure".
//!
//! Write your code in `src/denoise/exercise.rs`. Check it with `cargo test -p day05 denoise::`, and run it on
//! a test image with `cargo run -p day05 -- denoise` (pictures in `out/denoise/`).
//! `src/denoise/solution.rs` is a model solution: `cargo test -p day05 --features solution denoise::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day05 -- denoise`.
pub mod run;
