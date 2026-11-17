//! Coding exercises for the unit "Brightness constancy and the aperture problem".
//!
//! Write your code in `src/flow_constraint/exercise.rs`. Check it with `cargo test -p opticalflow flow_constraint::`, and run it on
//! the micrograph with `cargo run -p opticalflow -- flow-constraint` (pictures in `out/flow-constraint/`).
//! `src/flow_constraint/solution.rs` is a model solution: `cargo test -p opticalflow --features solution flow_constraint::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p opticalflow -- flow-constraint`.
pub mod run;
