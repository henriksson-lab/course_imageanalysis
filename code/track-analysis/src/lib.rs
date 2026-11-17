//! Coding exercises for the unit "Tracking: speed, MSD and modes of motion".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p track-analysis`, and run the
//! examples with `cargo run -p track-analysis` (pictures in `out/track-analysis/`).
//! `src/solution.rs` is a model solution: `cargo test -p track-analysis --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
