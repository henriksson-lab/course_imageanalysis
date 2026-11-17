//! Coding exercises for the unit "Stain separation: colour deconvolution".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p stain-unmixing`, and run it
//! with `cargo run -p stain-unmixing` (pictures in `out/stain-unmixing/`).
//! `src/solution.rs` is a model solution: `cargo test -p stain-unmixing --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
