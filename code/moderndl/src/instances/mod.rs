//! Coding exercises for the unit "From masks to objects".
//!
//! Write your code in `src/instances/exercise.rs`. Check it with `cargo test -p moderndl instances::`, and run it
//! with `cargo run -p moderndl -- instances` (pictures in `out/instances/`).
//! `src/instances/solution.rs` is a model solution: `cargo test -p moderndl --features solution instances::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p moderndl -- instances`.
pub mod run;
