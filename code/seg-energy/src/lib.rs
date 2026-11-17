//! Coding exercises for the unit "Segmentation as energy minimisation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p seg-energy`, and run the
//! examples with `cargo run -p seg-energy` (pictures in `out/seg-energy/`).
//! `src/solution.rs` is a model solution: `cargo test -p seg-energy --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
