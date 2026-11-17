//! Coding exercises for the unit "Many views: structure from motion and bundle adjustment".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p sfm`, and run it on
//! a flatland scene with `cargo run -p sfm` (pictures in `out/sfm/`).
//! `src/solution.rs` is a model solution: `cargo test -p sfm --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
