//! Coding exercises for the unit "Shadows, mirrors and glass: recursive ray tracing".
//!
//! Write your code in `src/whitted/exercise.rs`. Check it with `cargo test -p day50 whitted::`, and run it on
//! a small scene with `cargo run -p day50 -- whitted` (pictures in `out/whitted/`).
//! `src/whitted/solution.rs` is a model solution: `cargo test -p day50 --features solution whitted::`.
//! `src/scene.rs` holds the scene and the ray casting of the last unit, ready to use.
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

/// The program of this unit: `cargo run -p day50 -- whitted`.
pub mod run;
