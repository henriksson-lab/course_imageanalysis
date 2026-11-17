//! Coding exercises for the unit "The pinhole camera and calibration".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p cameras`, and run it on
//! a synthetic calibration target with `cargo run -p cameras` (pictures in `out/cameras/`).
//! `src/solution.rs` is a model solution: `cargo test -p cameras --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
