//! Coding exercises for the unit "FRET in pictures: a ruler for nanometres".
//!
//! Write your code in `src/fret_physics/exercise.rs`. Check it with `cargo test -p day45 fret_physics::`, and run it with
//! `cargo run -p day45 -- fret-physics` (results in `out/fret-physics/`).
//! `src/fret_physics/solution.rs` is a model solution: `cargo test -p day45 --features solution fret_physics::`.
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

/// The program of this unit: `cargo run -p day45 -- fret-physics`.
pub mod run;
