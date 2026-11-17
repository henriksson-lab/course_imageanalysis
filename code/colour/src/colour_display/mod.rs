//! Coding exercises for the unit "Colour for display: colour maps, merges and decorrelation".
//!
//! Write your code in `src/colour_display/exercise.rs`. Check it with `cargo test -p colour colour_display::`, and run it
//! with `cargo run -p colour -- colour-display` (pictures in `out/colour-display/`).
//! `src/colour_display/solution.rs` is a model solution: `cargo test -p colour --features solution colour_display::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p colour -- colour-display`.
pub mod run;
