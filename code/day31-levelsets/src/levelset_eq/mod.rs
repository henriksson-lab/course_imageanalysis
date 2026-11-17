//! Coding exercises for the unit "Moving a curve: the level set equation".
//!
//! Write your code in `src/levelset_eq/exercise.rs`. Check it with `cargo test -p day31 levelset_eq::`, and run it
//! with `cargo run -p day31 -- levelset-eq` (pictures in `out/levelset-eq/`).
//! `src/levelset_eq/solution.rs` is a model solution: `cargo test -p day31 --features solution levelset_eq::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day31 -- levelset-eq`.
pub mod run;
