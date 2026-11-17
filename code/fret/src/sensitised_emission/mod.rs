//! Coding exercises for the unit "Three images, two controls: correcting sensitised emission".
//!
//! Write your code in `src/sensitised_emission/exercise.rs`. Check it with `cargo test -p fret sensitised_emission::`, and run it with
//! `cargo run -p fret -- sensitised-emission` (results in `out/sensitised-emission/`).
//! `src/sensitised_emission/solution.rs` is a model solution: `cargo test -p fret --features solution sensitised_emission::`.
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

/// The program of this unit: `cargo run -p fret -- sensitised-emission`.
pub mod run;
