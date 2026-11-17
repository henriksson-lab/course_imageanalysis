//! Coding exercises for the unit "Phase correlation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p phasecorr`, and run it on
//! the micrograph with `cargo run -p phasecorr` (pictures in `out/phasecorr/`).
//! `src/solution.rs` is a model solution: `cargo test -p phasecorr --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
