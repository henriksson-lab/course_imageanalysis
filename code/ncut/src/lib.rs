//! Coding exercises for the unit "Spectral partitioning and normalized cuts".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p ncut`, and run the
//! examples with `cargo run -p ncut` (pictures in `out/ncut/`).
//! `src/solution.rs` is a model solution: `cargo test -p ncut --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
