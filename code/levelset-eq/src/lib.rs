//! Coding exercises for the unit "Moving a curve: the level set equation".
//!
//! Write your code in `src/exercise.rs`. Check it with `cargo test -p levelset-eq`, and run it
//! with `cargo run -p levelset-eq` (pictures in `out/levelset-eq/`).
//! `src/solution.rs` is a model solution: `cargo test -p levelset-eq --features solution`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;
