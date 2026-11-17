//! Coding exercises for the unit "3D images".
//!
//! Write your code in `src/volumes3d/exercise.rs`. Check it with `cargo test -p day47 volumes3d::`, and run it
//! with `cargo run -p day47 -- volumes3d` (pictures in `out/volumes3d/`).
//! `src/volumes3d/solution.rs` is a model solution: `cargo test -p day47 --features solution volumes3d::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day47 -- volumes3d`.
pub mod run;
