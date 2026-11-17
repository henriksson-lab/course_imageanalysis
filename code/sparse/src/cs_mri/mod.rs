//! Coding exercises for the unit "Fast MRI with compressed sensing".
//!
//! Write your code in `src/cs_mri/exercise.rs`. Check it with `cargo test -p sparse cs_mri::`, and run it
//! on a head phantom with `cargo run -p sparse -- cs-mri` (pictures in `out/cs-mri/`).
//! `src/cs_mri/solution.rs` is a model solution: `cargo test -p sparse --features solution cs_mri::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p sparse -- cs-mri`.
pub mod run;
