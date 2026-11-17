//! Coding exercises for the unit "Endmembers: the geometry of mixtures".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p endmembers`, and run it
//! with `cargo run -p endmembers` (pictures in `out/endmembers/`).
//! `src/solution.rs` is a model solution: `cargo test -p endmembers --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
