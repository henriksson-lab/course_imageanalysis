//! Coding exercises for the unit "Meshes, coordinate spaces and projection".
//!
//! Write your code in `src/pipeline3d/exercise.rs`. Check it with `cargo test -p day48 pipeline3d::`, and run it on
//! a small scene of houses with `cargo run -p day48 -- pipeline3d` (pictures in `out/pipeline3d/`).
//! `src/pipeline3d/solution.rs` is a model solution: `cargo test -p day48 --features solution pipeline3d::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day48 -- pipeline3d`.
pub mod run;
