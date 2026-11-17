//! Coding exercises for the unit "Measuring surfaces".
//!
//! Write your code in `src/surface_measure/exercise.rs`. Check it with `cargo test -p marchingcubes surface_measure::`, and run it
//! with `cargo run -p marchingcubes -- surface-measure` (it prints its measurements).
//! `src/surface_measure/solution.rs` is a model solution: `cargo test -p marchingcubes --features solution surface_measure::`.
//! `src/surface.rs` makes the meshes to measure (not an exercise).

mod surface;
pub use surface::{surface, Mesh};

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p marchingcubes -- surface-measure`.
pub mod run;
