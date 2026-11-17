//! Coding exercises for the unit "12- and 16-bit images in practice".
//!
//! Write your code in `src/highbit/exercise.rs`. Check it with `cargo test -p day03 highbit::`, and run it
//! with `cargo run -p day03 -- highbit` (pictures in `out/highbit/`).
//! `src/highbit/solution.rs` is a model solution: `cargo test -p day03 --features solution highbit::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day03 -- highbit`.
pub mod run;
