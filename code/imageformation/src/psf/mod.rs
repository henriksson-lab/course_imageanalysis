//! Coding exercises for the unit "Image formation and the point spread function".
//!
//! Write your code in `src/psf/exercise.rs`. Check it with `cargo test -p imageformation psf::`, and run it on
//! the micrograph with `cargo run -p imageformation -- psf` (pictures in `out/psf/`).
//! `src/psf/solution.rs` is a model solution: `cargo test -p imageformation --features solution psf::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p imageformation -- psf`.
pub mod run;
