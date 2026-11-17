//! Coding exercises for the unit "Finding spots: filter, then local maxima".
//!
//! Write your code in `src/spot_filters/exercise.rs`. Check it with `cargo test -p pointdetection spot_filters::`, and run it
//! on a synthetic spot image with `cargo run -p pointdetection -- spot-filters` (pictures in `out/spot-filters/`).
//! `src/spot_filters/solution.rs` is a model solution: `cargo test -p pointdetection --features solution spot_filters::`.
//! `src/common.rs` makes the test images, `src/spot_filters/sim.rs` draws the results; they are not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p pointdetection -- spot-filters`.
pub mod run;
