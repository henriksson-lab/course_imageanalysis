//! Coding exercises for the unit "Rays from the camera, and where they hit".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p raycast`, and run it on
//! a small scene with `cargo run -p raycast` (pictures in `out/raycast/`).
//! `src/solution.rs` is a model solution: `cargo test -p raycast --features solution`.

mod vec3;
pub use vec3::*;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
