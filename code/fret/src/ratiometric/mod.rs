//! Coding exercises for the unit "Ratio imaging, biosensors and the noise of a ratio".
//!
//! Write your code in `src/ratiometric/exercise.rs`. Check it with `cargo test -p fret ratiometric::`, and run it with
//! `cargo run -p fret -- ratiometric` (results in `out/ratiometric/`).
//! `src/ratiometric/solution.rs` is a model solution: `cargo test -p fret --features solution ratiometric::`.
//! `src/sim.rs` makes the test data and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p fret -- ratiometric`.
pub mod run;
