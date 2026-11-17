//! Coding exercises for the unit "Matching objects: IoU, F1 and average precision".
//!
//! Write your code in `src/object_eval/exercise.rs`. Check it with `cargo test -p day37 object_eval::`, and run it
//! with `cargo run -p day37 -- object-eval` (pictures in `out/object-eval/`).
//! `src/object_eval/solution.rs` is a model solution: `cargo test -p day37 --features solution object_eval::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day37 -- object-eval`.
pub mod run;
