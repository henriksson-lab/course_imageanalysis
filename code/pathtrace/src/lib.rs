//! Coding exercises for the unit "Path tracing: the rendering equation, Monte Carlo and noise".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p pathtrace`, and run it
//! on a closed box with `cargo run -p pathtrace` (pictures in `out/pathtrace/`).
//! `src/solution.rs` is a model solution: `cargo test -p pathtrace --features solution`.
//! `src/scene.rs` holds the scene, the nearest hit and the camera, ready to use.

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
