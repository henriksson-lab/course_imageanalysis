//! Coding exercises for the unit "Reconstruction and h-extrema".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p morph-reconstruct`, and run it on
//! synthetic nuclei with `cargo run -p morph-reconstruct` (pictures in `out/morph-reconstruct/`).
//! `src/solution.rs` is a model solution: `cargo test -p morph-reconstruct --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
