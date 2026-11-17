//! Coding exercises for the unit "Compressed sensing: fewer measurements than unknowns".
//!
//! Write your code in `src/compressed_sensing/exercise.rs`. Check it with `cargo test -p day11 compressed_sensing::`, and run it
//! on sparse test signals with `cargo run -p day11 -- compressed-sensing` (pictures in `out/compressed-sensing/`).
//! `src/compressed_sensing/solution.rs` is a model solution: `cargo test -p day11 --features solution compressed_sensing::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day11 -- compressed-sensing`.
pub mod run;
