//! Coding exercises for the unit "Segmentation networks".
//!
//! Write your code in `src/unet/exercise.rs`. Check it with `cargo test -p day36 unet::`, and run it
//! with `cargo run -p day36 -- unet` (pictures in `out/unet/`).
//! `src/unet/solution.rs` is a model solution: `cargo test -p day36 --features solution unet::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day36 -- unet`.
pub mod run;
