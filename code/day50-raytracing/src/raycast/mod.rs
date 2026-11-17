//! Coding exercises for the unit "Rays from the camera, and where they hit".
//!
//! Write your code in `src/raycast/exercise.rs`. Check it with `cargo test -p day50 raycast::`, and run it on
//! a small scene with `cargo run -p day50 -- raycast` (pictures in `out/raycast/`).
//! `src/raycast/solution.rs` is a model solution: `cargo test -p day50 --features solution raycast::`.

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

/// The program of this unit: `cargo run -p day50 -- raycast`.
pub mod run;
