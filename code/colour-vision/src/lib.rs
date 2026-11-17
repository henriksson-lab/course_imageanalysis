//! Coding exercises for the unit "From a spectrum to three numbers".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p colour-vision`, and run it
//! with `cargo run -p colour-vision` (pictures in `out/colour-vision/`).
//! `src/solution.rs` is a model solution: `cargo test -p colour-vision --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
