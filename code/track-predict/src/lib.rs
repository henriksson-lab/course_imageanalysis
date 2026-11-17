//! Coding exercises for the unit "Tracking: motion models and the Kalman filter".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p track-predict`, and run the
//! examples with `cargo run -p track-predict` (pictures in `out/track-predict/`).
//! `src/solution.rs` is a model solution: `cargo test -p track-predict --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
