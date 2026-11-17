//! Coding exercises for the unit "The pinhole camera and calibration".
//!
//! Write your code in `src/cameras/exercise.rs`. Check it with `cargo test -p day49 cameras::`, and run it on
//! a synthetic calibration target with `cargo run -p day49 -- cameras` (pictures in `out/cameras/`).
//! `src/cameras/solution.rs` is a model solution: `cargo test -p day49 --features solution cameras::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day49 -- cameras`.
pub mod run;
