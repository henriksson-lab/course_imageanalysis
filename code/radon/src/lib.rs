//! Coding exercises for the unit "Projections and the sinogram".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p radon`, and run it on
//! test images with `cargo run -p radon` (pictures in `out/radon/`).
//! `src/solution.rs` is a model solution: `cargo test -p radon --features solution`.

pub mod tomo;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
