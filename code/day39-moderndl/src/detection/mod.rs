//! Coding exercises for the unit "Finding objects as boxes".
//!
//! Write your code in `src/detection/exercise.rs`. Check it with `cargo test -p day39 detection::`, and run it
//! with `cargo run -p day39 -- detection` (pictures in `out/detection/`).
//! `src/detection/solution.rs` is a model solution: `cargo test -p day39 --features solution detection::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day39 -- detection`.
pub mod run;
