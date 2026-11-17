//! Coding exercises for the unit "Frequency domain and phasors".
//!
//! Write your code in `src/phasor/exercise.rs`. Check it with `cargo test -p flim phasor::`, and run it
//! with `cargo run -p flim -- phasor` (pictures in `out/phasor/`).
//! `src/phasor/solution.rs` is a model solution: `cargo test -p flim --features solution phasor::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p flim -- phasor`.
pub mod run;
