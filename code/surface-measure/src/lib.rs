//! Coding exercises for the unit "Measuring surfaces".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p surface-measure`, and run it
//! with `cargo run -p surface-measure` (it prints its measurements).
//! `src/solution.rs` is a model solution: `cargo test -p surface-measure --features solution`.
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
