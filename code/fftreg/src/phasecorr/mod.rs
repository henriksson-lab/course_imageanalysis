//! Coding exercises for the unit "Phase correlation".
//!
//! Write your code in `src/phasecorr/exercise.rs`. Check it with `cargo test -p fftreg phasecorr::`, and run it on
//! the micrograph with `cargo run -p fftreg -- phasecorr` (pictures in `out/phasecorr/`).
//! `src/phasecorr/solution.rs` is a model solution: `cargo test -p fftreg --features solution phasecorr::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p fftreg -- phasecorr`.
pub mod run;
