//! Coding exercises for the unit "Finding spots: filter, then local maxima".
//!
//! Write your code in `src/spot_filters/exercise.rs`. Check it with `cargo test -p day20 spot_filters::`, and run it
//! on a synthetic spot image with `cargo run -p day20 -- spot-filters` (pictures in `out/spot-filters/`).
//! `src/spot_filters/solution.rs` is a model solution: `cargo test -p day20 --features solution spot_filters::`.
//! `src/sim.rs` makes the test images and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day20 -- spot-filters`.
pub mod run;
