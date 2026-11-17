//! Coding exercises for the unit "3D images".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p volumes3d`, and run it
//! with `cargo run -p volumes3d` (pictures in `out/volumes3d/`).
//! `src/solution.rs` is a model solution: `cargo test -p volumes3d --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
