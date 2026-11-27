//! Coding exercises for the unit "Outlines as periodic signals: Fourier descriptors".
//!
//! Write your code in `src/fourier_descriptors/exercise.rs`. Check it with `cargo test -p shapematch fourier_descriptors::`, and run it on
//! the test shapes with `cargo run -p shapematch -- fourier-descriptors` (pictures in `out/fourier-descriptors/`).
//! `src/fourier_descriptors/solution.rs` is a model solution: `cargo test -p shapematch --features solution fourier_descriptors::`.
//! `src/fourier_descriptors/given.rs` has the boundary tracing, polygon filling, test shapes and drawing, ready to use.

pub mod given;
pub use given::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p shapematch -- fourier-descriptors`.
pub mod run;
