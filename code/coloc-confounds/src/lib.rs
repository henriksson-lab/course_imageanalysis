//! Coding exercises for the unit "What the microscope does to colocalisation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p coloc-confounds`, and run it with
//! `cargo run -p coloc-confounds` (results in `out/coloc-confounds/`).
//! `src/solution.rs` is a model solution: `cargo test -p coloc-confounds --features solution`.
//! `src/sim.rs` makes the test data and draws the results; it is not part of the exercise.

pub mod sim;

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
