//! Coding exercises for the unit "Tracking: motion models and the Kalman filter".
//!
//! Write your code in `src/track_predict/exercise.rs`. Check it with `cargo test -p tracking track_predict::`, and run the
//! examples with `cargo run -p tracking -- track-predict` (pictures in `out/track-predict/`).
//! `src/track_predict/solution.rs` is a model solution: `cargo test -p tracking --features solution track_predict::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p tracking -- track-predict`.
pub mod run;
