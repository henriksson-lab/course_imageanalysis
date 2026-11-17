//! Coding exercises for the unit "From a spectrum to three numbers".
//!
//! Write your code in `src/colour_vision/exercise.rs`. Check it with `cargo test -p colour colour_vision::`, and run it
//! with `cargo run -p colour -- colour-vision` (pictures in `out/colour-vision/`).
//! `src/colour_vision/solution.rs` is a model solution: `cargo test -p colour --features solution colour_vision::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p colour -- colour-vision`.
pub mod run;
