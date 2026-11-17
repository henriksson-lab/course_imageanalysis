//! Coding exercises for the unit "From measurements to claims".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p replicates`, and run it
//! with `cargo run -p replicates` (pictures in `out/replicates/`).
//! `src/solution.rs` is a model solution: `cargo test -p replicates --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
