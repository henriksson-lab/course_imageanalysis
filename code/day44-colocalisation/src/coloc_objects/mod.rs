//! Coding exercises for the unit "Objects instead of pixels: overlap, distances and the chance level".
//!
//! Write your code in `src/coloc_objects/exercise.rs`. Check it with `cargo test -p day44 coloc_objects::`, and run it with
//! `cargo run -p day44 -- coloc-objects` (results in `out/coloc-objects/`).
//! `src/coloc_objects/solution.rs` is a model solution: `cargo test -p day44 --features solution coloc_objects::`.
//! `src/coloc_objects/sim.rs` makes the test data and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day44 -- coloc-objects`.
pub mod run;
