//! Coding exercises for the unit "Shape models: landmarks, Procrustes and PCA".
//!
//! Write your code in `src/shape_models/exercise.rs`. Check it with `cargo test -p day30 shape_models::`, and run it
//! with `cargo run -p day30 -- shape-models` (pictures in `out/shape-models/`).
//! `src/shape_models/solution.rs` is a model solution: `cargo test -p day30 --features solution shape_models::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day30 -- shape-models`.
pub mod run;
