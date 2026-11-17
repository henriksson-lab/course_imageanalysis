//! Coding exercises for the unit "Stain separation: colour deconvolution".
//!
//! Write your code in `src/stain_unmixing/exercise.rs`. Check it with `cargo test -p day43 stain_unmixing::`, and run it
//! with `cargo run -p day43 -- stain-unmixing` (pictures in `out/stain-unmixing/`).
//! `src/stain_unmixing/solution.rs` is a model solution: `cargo test -p day43 --features solution stain_unmixing::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day43 -- stain-unmixing`.
pub mod run;
