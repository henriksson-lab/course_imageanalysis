//! Coding exercises for the unit "Resolution, sampling and measuring the PSF".
//!
//! Write your code in `src/resolution/exercise.rs`. Check it with `cargo test -p imageformation resolution::`, and run it on
//! a simulated bead slide with `cargo run -p imageformation -- resolution` (pictures in `out/resolution/`).
//! `src/resolution/solution.rs` is a model solution: `cargo test -p imageformation --features solution resolution::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p imageformation -- resolution`.
pub mod run;
