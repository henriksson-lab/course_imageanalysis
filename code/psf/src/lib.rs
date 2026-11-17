//! Coding exercises for the unit "Image formation and the point spread function".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p psf`, and run it on
//! the micrograph with `cargo run -p psf` (pictures in `out/psf/`).
//! `src/solution.rs` is a model solution: `cargo test -p psf --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
