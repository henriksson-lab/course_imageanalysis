//! Coding exercises for the unit "How graphics cards store colour".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p gpuformats`, and run it on
//! the course micrograph with `cargo run -p gpuformats` (pictures in `out/gpuformats/`).
//! `src/solution.rs` is a model solution: `cargo test -p gpuformats --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
