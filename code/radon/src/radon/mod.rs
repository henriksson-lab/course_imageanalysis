//! Coding exercises for the unit "Projections and the sinogram".
//!
//! Write your code in `src/radon/exercise.rs`. Check it with `cargo test -p radon radon::`, and run it on
//! test images with `cargo run -p radon -- radon` (pictures in `out/radon/`).
//! `src/radon/solution.rs` is a model solution: `cargo test -p radon --features solution radon::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p radon -- radon`.
pub mod run;
