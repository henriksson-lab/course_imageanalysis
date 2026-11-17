//! Coding exercises for the unit "Image-to-image translation and hallucination".
//!
//! Write your code in `src/image_translation/exercise.rs`. Check it with `cargo test -p gans image_translation::`, and run it
//! with `cargo run -p gans -- image-translation` (pictures in `out/image-translation/`).
//! `src/image_translation/solution.rs` is a model solution: `cargo test -p gans --features solution image_translation::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p gans -- image-translation`.
pub mod run;
