//! Coding exercises for the unit "Meshes, coordinate spaces and projection".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p pipeline3d`, and run it on
//! a small scene of houses with `cargo run -p pipeline3d` (pictures in `out/pipeline3d/`).
//! `src/solution.rs` is a model solution: `cargo test -p pipeline3d --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
