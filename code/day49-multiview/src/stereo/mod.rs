//! Coding exercises for the unit "Stereo: from disparity to depth".
//!
//! Write your code in `src/stereo/exercise.rs`. Check it with `cargo test -p day49 stereo::`, and run it on
//! a synthetic stereo pair with `cargo run -p day49 -- stereo` (pictures in `out/stereo/`).
//! `src/stereo/solution.rs` is a model solution: `cargo test -p day49 --features solution stereo::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day49 -- stereo`.
pub mod run;
