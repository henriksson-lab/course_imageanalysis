//! Coding exercises for the unit "Keypoints: corners and blobs across scale".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p keypoints`, and run it on
//! the micrograph with `cargo run -p keypoints` (pictures in `out/keypoints/`).
//! `src/solution.rs` is a model solution: `cargo test -p keypoints --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
