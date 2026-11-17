//! Coding exercises for the unit "Segmentation as energy minimisation".
//!
//! Write your code in `src/seg_energy/exercise.rs`. Check it with `cargo test -p graphcuts seg_energy::`, and run the
//! examples with `cargo run -p graphcuts -- seg-energy` (pictures in `out/seg-energy/`).
//! `src/seg_energy/solution.rs` is a model solution: `cargo test -p graphcuts --features solution seg_energy::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p graphcuts -- seg-energy`.
pub mod run;
