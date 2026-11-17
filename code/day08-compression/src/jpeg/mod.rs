//! Coding exercises for the unit "The JPEG pipeline, artefacts and quality metrics".
//!
//! Write your code in `src/jpeg/exercise.rs`. Check it with `cargo test -p day08 jpeg::`, and run it on
//! the micrograph with `cargo run -p day08 -- jpeg` (pictures in `out/jpeg/`).
//! `src/jpeg/solution.rs` is a model solution: `cargo test -p day08 --features solution jpeg::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day08 -- jpeg`.
pub mod run;
