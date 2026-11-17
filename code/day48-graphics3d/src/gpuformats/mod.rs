//! Coding exercises for the unit "How graphics cards store colour".
//!
//! Write your code in `src/gpuformats/exercise.rs`. Check it with `cargo test -p day48 gpuformats::`, and run it on
//! the course micrograph with `cargo run -p day48 -- gpuformats` (pictures in `out/gpuformats/`).
//! `src/gpuformats/solution.rs` is a model solution: `cargo test -p day48 --features solution gpuformats::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day48 -- gpuformats`.
pub mod run;
