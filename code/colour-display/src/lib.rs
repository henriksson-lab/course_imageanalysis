//! Coding exercises for the unit "Colour for display: colour maps, merges and decorrelation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p colour-display`, and run it
//! with `cargo run -p colour-display` (pictures in `out/colour-display/`).
//! `src/solution.rs` is a model solution: `cargo test -p colour-display --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
