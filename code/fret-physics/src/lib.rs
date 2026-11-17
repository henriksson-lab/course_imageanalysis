//! Coding exercises for the unit "FRET in pictures: a ruler for nanometres".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p fret-physics`, and run it with
//! `cargo run -p fret-physics` (results in `out/fret-physics/`).
//! `src/solution.rs` is a model solution: `cargo test -p fret-physics --features solution`.
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
