//! Coding exercises for the unit "Crowded spots and evaluating a spot detector".
//!
//! Write your code in `src/spot_crowd/exercise.rs`. Check it with `cargo test -p day20 spot_crowd::`, and run it
//! on synthetic spot images with `cargo run -p day20 -- spot-crowd` (pictures in `out/spot-crowd/`).
//! `src/spot_crowd/solution.rs` is a model solution: `cargo test -p day20 --features solution spot_crowd::`.
//! `src/common.rs` makes the test images, `src/spot_crowd/sim.rs` draws the results; they are not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day20 -- spot-crowd`.
pub mod run;
