//! Coding exercises for the unit "Watershed segmentation of a real image".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ws-gradient`, and run it on
//! the micrograph with `cargo run -p ws-gradient` (pictures in `out/ws-gradient/`).
//! `src/solution.rs` is a model solution: `cargo test -p ws-gradient --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
