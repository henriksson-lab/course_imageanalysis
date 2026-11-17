//! Coding exercises for the unit "Three images, two controls: correcting sensitised emission".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p sensitised-emission`, and run it with
//! `cargo run -p sensitised-emission` (results in `out/sensitised-emission/`).
//! `src/solution.rs` is a model solution: `cargo test -p sensitised-emission --features solution`.
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
