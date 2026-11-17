//! Coding exercises for the unit "Resolution, sampling and measuring the PSF".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p resolution`, and run it on
//! a simulated bead slide with `cargo run -p resolution` (pictures in `out/resolution/`).
//! `src/solution.rs` is a model solution: `cargo test -p resolution --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
