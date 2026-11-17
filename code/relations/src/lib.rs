//! Coding exercises for the unit "Neighbours, parents and children".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p relations`, and run it with
//! `cargo run -p relations` (pictures in `out/relations/`).
//! `src/solution.rs` is a model solution: `cargo test -p relations --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
