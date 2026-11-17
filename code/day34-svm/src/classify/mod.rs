//! Coding exercises for the unit "Linear classifiers".
//!
//! Write your code in `src/classify/exercise.rs`. Check it with `cargo test -p day34 classify::`, and run it on
//! examples with `cargo run -p day34 -- classify` (pictures in `out/classify/`).
//! `src/classify/solution.rs` is a model solution: `cargo test -p day34 --features solution classify::`.

#[cfg(not(feature = "solution"))]
mod exercise;
#[cfg(not(feature = "solution"))]
pub use exercise::*;

#[cfg(feature = "solution")]
mod solution;
#[cfg(feature = "solution")]
pub use solution::*;

/// The program of this unit: `cargo run -p day34 -- classify`.
pub mod run;
