//! Coding exercises for the unit "Scale space, blobs and pyramids".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p scale-space`, and run it
//! on the test images with `cargo run -p scale-space` (pictures in `out/scale-space/`).
//! `src/solution.rs` is a model solution: `cargo test -p scale-space --features solution`.
//! `src/given.rs` has the Gaussian blur of the edges unit, ready to use.

pub mod given;
pub use given::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
