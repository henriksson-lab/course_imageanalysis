//! Coding exercises for the unit "Path tracing: the rendering equation, Monte Carlo and noise".
//!
//! Write your code in `src/pathtrace/exercise.rs`. Check it with `cargo test -p raytracing pathtrace::`, and run it
//! on a closed box with `cargo run -p raytracing -- pathtrace` (pictures in `out/pathtrace/`).
//! `src/pathtrace/solution.rs` is a model solution: `cargo test -p raytracing --features solution pathtrace::`.
//! `src/scene.rs` holds the scene, the nearest hit and the camera, ready to use.
//! The 3D vectors and rays are in `src/common.rs`.

mod scene;
pub use crate::common::*;
pub use scene::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p raytracing -- pathtrace`.
pub mod run;
