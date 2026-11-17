//! Coding exercises for the unit "Rays from the camera, and where they hit".
//!
//! Write your code in `src/raycast/exercise.rs`. Check it with `cargo test -p raytracing raycast::`, and run it on
//! a small scene with `cargo run -p raytracing -- raycast` (pictures in `out/raycast/`).
//! `src/raycast/solution.rs` is a model solution: `cargo test -p raytracing --features solution raycast::`.
//! The 3D vectors and rays are in `src/common.rs`.

pub use crate::common::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p raytracing -- raycast`.
pub mod run;
