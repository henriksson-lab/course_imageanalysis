//! Coding exercises for the unit "From measurements to claims".
//!
//! Write your code in `src/replicates/exercise.rs`. Check it with `cargo test -p day37 replicates::`, and run it
//! with `cargo run -p day37 -- replicates` (pictures in `out/replicates/`).
//! `src/replicates/solution.rs` is a model solution: `cargo test -p day37 --features solution replicates::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day37 -- replicates`.
pub mod run;
