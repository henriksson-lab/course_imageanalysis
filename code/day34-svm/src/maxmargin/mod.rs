//! Coding exercises for the unit "Maximum margin and support vectors".
//!
//! Write your code in `src/maxmargin/exercise.rs`. Check it with `cargo test -p day34 maxmargin::`, and run it on
//! examples with `cargo run -p day34 -- maxmargin` (pictures in `out/maxmargin/`).
//! `src/maxmargin/solution.rs` is a model solution: `cargo test -p day34 --features solution maxmargin::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day34 -- maxmargin`.
pub mod run;
