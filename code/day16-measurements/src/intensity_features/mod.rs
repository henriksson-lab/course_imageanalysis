//! Coding exercises for the unit "Intensity inside an object".
//!
//! Write your code in `src/intensity_features/exercise.rs`. Check it with `cargo test -p day16 intensity_features::`, and run it with
//! `cargo run -p day16 -- intensity-features` (pictures in `out/intensity-features/`).
//! `src/intensity_features/solution.rs` is a model solution: `cargo test -p day16 --features solution intensity_features::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day16 -- intensity-features`.
pub mod run;
