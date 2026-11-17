//! Coding exercises for the unit "Phase correlation".
//!
//! Write your code in `src/phasecorr/exercise.rs`. Check it with `cargo test -p day23 phasecorr::`, and run it on
//! the micrograph with `cargo run -p day23 -- phasecorr` (pictures in `out/phasecorr/`).
//! `src/phasecorr/solution.rs` is a model solution: `cargo test -p day23 --features solution phasecorr::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day23 -- phasecorr`.
pub mod run;
