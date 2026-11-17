//! Coding exercises for the unit "FLIM-FRET and metabolic imaging".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p flim-fret`, and run it
//! with `cargo run -p flim-fret` (pictures in `out/flim-fret/`).
//! `src/solution.rs` is a model solution: `cargo test -p flim-fret --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
