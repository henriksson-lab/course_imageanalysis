//! Coding exercises for the unit "Many views: structure from motion and bundle adjustment".
//!
//! Write your code in `src/sfm/exercise.rs`. Check it with `cargo test -p day49 sfm::`, and run it on
//! a flatland scene with `cargo run -p day49 -- sfm` (pictures in `out/sfm/`).
//! `src/sfm/solution.rs` is a model solution: `cargo test -p day49 --features solution sfm::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day49 -- sfm`.
pub mod run;
