//! Coding exercises for the unit "Shape features".
//!
//! Write your code in `src/shape_features/exercise.rs`. Check it with `cargo test -p day14 shape_features::`, and run it on
//! synthetic nuclei with `cargo run -p day14 -- shape-features` (pictures in `out/shape-features/`).
//! `src/shape_features/solution.rs` is a model solution: `cargo test -p day14 --features solution shape_features::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day14 -- shape-features`.
pub mod run;
