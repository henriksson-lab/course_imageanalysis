//! Coding exercises for the unit "Shadows, mirrors and glass: recursive ray tracing".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p whitted`, and run it on
//! a small scene with `cargo run -p whitted` (pictures in `out/whitted/`).
//! `src/solution.rs` is a model solution: `cargo test -p whitted --features solution`.
//! `src/scene.rs` holds the scene and the ray casting of the last unit, ready to use.

mod scene;
mod vec3;
pub use scene::*;
pub use vec3::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
