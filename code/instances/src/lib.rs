//! Coding exercises for the unit "From masks to objects".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p instances`, and run it
//! with `cargo run -p instances` (pictures in `out/instances/`).
//! `src/solution.rs` is a model solution: `cargo test -p instances --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
