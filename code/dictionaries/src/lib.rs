//! Coding exercises for the unit "Sparse coding and dictionary learning".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p dictionaries`, and run it on
//! the micrograph with `cargo run -p dictionaries` (pictures in `out/dictionaries/`).
//! `src/solution.rs` is a model solution: `cargo test -p dictionaries --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
