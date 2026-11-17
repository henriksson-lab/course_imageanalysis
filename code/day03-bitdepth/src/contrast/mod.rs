//! Coding exercises for the unit "Contrast and intensity transforms".
//!
//! Write your code in `src/contrast/exercise.rs`. Check it with `cargo test -p day03 contrast::`, and run it
//! with `cargo run -p day03 -- contrast` (pictures in `out/contrast/`).
//! `src/contrast/solution.rs` is a model solution: `cargo test -p day03 --features solution contrast::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day03 -- contrast`.
pub mod run;
