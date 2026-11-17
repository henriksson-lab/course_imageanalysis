//! Coding exercises for the unit "12- and 16-bit images in practice".
//!
//! Write your code in `src/highbit/exercise.rs`. Check it with `cargo test -p bitdepth highbit::`, and run it
//! with `cargo run -p bitdepth -- highbit` (pictures in `out/highbit/`).
//! `src/highbit/solution.rs` is a model solution: `cargo test -p bitdepth --features solution highbit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p bitdepth -- highbit`.
pub mod run;
