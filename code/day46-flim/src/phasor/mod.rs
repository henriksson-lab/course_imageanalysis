//! Coding exercises for the unit "Frequency domain and phasors".
//!
//! Write your code in `src/phasor/exercise.rs`. Check it with `cargo test -p day46 phasor::`, and run it
//! with `cargo run -p day46 -- phasor` (pictures in `out/phasor/`).
//! `src/phasor/solution.rs` is a model solution: `cargo test -p day46 --features solution phasor::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day46 -- phasor`.
pub mod run;
