//! Coding exercises for the unit "FLIM-FRET and metabolic imaging".
//!
//! Write your code in `src/flim_fret/exercise.rs`. Check it with `cargo test -p day46 flim_fret::`, and run it
//! with `cargo run -p day46 -- flim-fret` (pictures in `out/flim-fret/`).
//! `src/flim_fret/solution.rs` is a model solution: `cargo test -p day46 --features solution flim_fret::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day46 -- flim-fret`.
pub mod run;
