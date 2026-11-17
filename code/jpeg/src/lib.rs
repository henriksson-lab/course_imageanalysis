//! Coding exercises for the unit "The JPEG pipeline, artefacts and quality metrics".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p jpeg`, and run it on
//! the micrograph with `cargo run -p jpeg` (pictures in `out/jpeg/`).
//! `src/solution.rs` is a model solution: `cargo test -p jpeg --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
