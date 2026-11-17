//! Coding exercises for the unit "Fast MRI with compressed sensing".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p cs-mri`, and run it
//! on a head phantom with `cargo run -p cs-mri` (pictures in `out/cs-mri/`).
//! `src/solution.rs` is a model solution: `cargo test -p cs-mri --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
